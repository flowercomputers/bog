//! Derive macros for fold3.
//!
//! - [`Codec`](macro@Codec): the positional binary encoding of a struct
//!   or enum, for keys and values of stored keyspaces: `Codec`,
//!   `OrderedCodec` and `ResidentSize` at once. **Field order is the
//!   schema and the sort order**: fields encode one after another with
//!   no names or tags, and a key struct sorts by its leading fields.
//! - [`Schema`](macro@Schema): the faces of a database from a struct of
//!   sources and views.
//!
//! Generated code refers to the library as `::fold3`. If the crate is
//! renamed or re-exported, point the derives at it with
//! `#[fold3(crate = "path::to::fold3")]` on the deriving type.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as Ts;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Path, parse_macro_input, parse_quote};

fn compile_error(span: proc_macro2::Span, msg: &str) -> TokenStream {
    syn::Error::new(span, msg).to_compile_error().into()
}

/// The path generated code uses to reach the library: `::fold3` unless
/// the deriving type overrides it with `#[fold3(crate = "path")]`.
fn crate_path(input: &DeriveInput) -> syn::Result<Path> {
    let mut path = None;
    for attr in &input.attrs {
        if attr.path().is_ident("fold3") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("crate") {
                    let lit: syn::LitStr = meta.value()?.parse()?;
                    path = Some(lit.parse::<Path>()?);
                    Ok(())
                } else {
                    Err(meta.error("unsupported fold3 attribute; expected `crate = \"path\"`"))
                }
            })?;
        }
    }
    Ok(path.unwrap_or_else(|| parse_quote!(::fold3)))
}

/// Destructuring pattern and bound names for `fields`.
fn bind(fields: &Fields) -> (Ts, Vec<syn::Ident>) {
    match fields {
        Fields::Named(f) => {
            let names: Vec<_> = f.named.iter().map(|f| f.ident.clone().unwrap()).collect();
            (quote! { { #( #names, )* } }, names)
        }
        Fields::Unnamed(f) => {
            let names: Vec<_> = (0..f.unnamed.len())
                .map(|i| format_ident!("f{i}"))
                .collect();
            (quote! { ( #( #names, )* ) }, names)
        }
        Fields::Unit => (Ts::new(), Vec::new()),
    }
}

/// Constructor from bound names: `{ a, .. }`, `(f0, ..)` or nothing.
fn construct(fields: &Fields, names: &[syn::Ident]) -> Ts {
    match fields {
        Fields::Named(_) => quote! { { #( #names, )* } },
        Fields::Unnamed(_) => quote! { ( #( #names, )* ) },
        Fields::Unit => Ts::new(),
    }
}

/// The three codec impls for one shape: `encode`, `decode`, `size` as
/// expressions over the bound names, for the plain and ordered codecs.
struct Shape {
    pattern: Ts,
    names: Vec<syn::Ident>,
    types: Vec<syn::Type>,
    ctor: Ts,
}

fn shape(fields: &Fields) -> Shape {
    let (pattern, names) = bind(fields);
    let types = fields.iter().map(|f| f.ty.clone()).collect();
    let ctor = construct(fields, &names);
    Shape {
        pattern,
        names,
        types,
        ctor,
    }
}

/// Positional binary encoding for keys and values: `Codec`,
/// `OrderedCodec` and `ResidentSize`. Fields encode in declaration
/// order; enum variants carry a one-byte tag in declaration order.
///
/// The layout carries no names: renaming fields is free, but reordering,
/// retyping, adding or removing them changes the on-disk format of any
/// keyspace using the type. Type parameters are bound by `Codec` and
/// `OrderedCodec`.
#[proc_macro_derive(Codec, attributes(fold3))]
pub fn derive_codec(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let krate = match crate_path(&input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };
    let codec = quote!(#krate::store::codec);

    let mut generics = input.generics.clone();
    for p in generics.type_params_mut() {
        p.bounds.push(parse_quote!(#codec::Codec));
        p.bounds.push(parse_quote!(#codec::OrderedCodec));
    }
    let (impl_g, ty_g, where_c) = generics.split_for_impl();

    // the body of each of the seven methods
    let (encode, encode_ordered, size, size_ordered, resident, decode, decode_ordered);
    match &input.data {
        Data::Struct(s) => {
            let Shape {
                pattern,
                names,
                types,
                ctor,
            } = shape(&s.fields);
            encode = quote! { let Self #pattern = self; #( <#types as #codec::Codec>::encode(#names, __out); )* };
            encode_ordered = quote! { let Self #pattern = self; #( <#types as #codec::OrderedCodec>::encode_ordered(#names, __out); )* };
            size = quote! { let Self #pattern = self; 0 #( + <#types as #codec::Codec>::encode_size(#names) )* };
            size_ordered = quote! { let Self #pattern = self; 0 #( + <#types as #codec::OrderedCodec>::encode_ordered_size(#names) )* };
            resident = quote! { let Self #pattern = self; ::std::mem::size_of::<Self>() #( + #codec::heap_size(#names) )* };
            decode = quote! {
                #( let #names = <#types as #codec::Codec>::decode(__input)?; )*
                ::std::result::Result::Ok(Self #ctor)
            };
            decode_ordered = quote! {
                #( let #names = <#types as #codec::OrderedCodec>::decode_ordered(__input)?; )*
                ::std::result::Result::Ok(Self #ctor)
            };
        }
        Data::Enum(e) => {
            if e.variants.len() > 256 {
                return compile_error(name.span(), "Codec enums support at most 256 variants");
            }
            let mut enc_arms = Ts::new();
            let mut enc_ord_arms = Ts::new();
            let mut size_arms = Ts::new();
            let mut size_ord_arms = Ts::new();
            let mut res_arms = Ts::new();
            let mut dec_arms = Ts::new();
            let mut dec_ord_arms = Ts::new();
            for (i, v) in e.variants.iter().enumerate() {
                let tag = i as u8;
                let vname = &v.ident;
                let Shape {
                    pattern,
                    names,
                    types,
                    ctor,
                } = shape(&v.fields);
                enc_arms.extend(quote! {
                    Self::#vname #pattern => { <u8 as #codec::Codec>::encode(&#tag, __out); #( <#types as #codec::Codec>::encode(#names, __out); )* }
                });
                enc_ord_arms.extend(quote! {
                    Self::#vname #pattern => { <u8 as #codec::Codec>::encode(&#tag, __out); #( <#types as #codec::OrderedCodec>::encode_ordered(#names, __out); )* }
                });
                size_arms.extend(quote! {
                    Self::#vname #pattern => 1 #( + <#types as #codec::Codec>::encode_size(#names) )*,
                });
                size_ord_arms.extend(quote! {
                    Self::#vname #pattern => 1 #( + <#types as #codec::OrderedCodec>::encode_ordered_size(#names) )*,
                });
                res_arms.extend(quote! {
                    Self::#vname #pattern => ::std::mem::size_of::<Self>() #( + #codec::heap_size(#names) )*,
                });
                dec_arms.extend(quote! {
                    #tag => { #( let #names = <#types as #codec::Codec>::decode(__input)?; )* ::std::result::Result::Ok(Self::#vname #ctor) }
                });
                dec_ord_arms.extend(quote! {
                    #tag => { #( let #names = <#types as #codec::OrderedCodec>::decode_ordered(__input)?; )* ::std::result::Result::Ok(Self::#vname #ctor) }
                });
            }
            let what = name.to_string();
            encode = quote! { match self { #enc_arms } };
            encode_ordered = quote! { match self { #enc_ord_arms } };
            size = quote! { match self { #size_arms } };
            size_ordered = quote! { match self { #size_ord_arms } };
            resident = quote! { match self { #res_arms } };
            decode = quote! {
                match <u8 as #codec::Codec>::decode(__input)? {
                    #dec_arms
                    tag => ::std::result::Result::Err(#codec::DecodeError::InvalidTag { what: #what, tag }),
                }
            };
            decode_ordered = quote! {
                match <u8 as #codec::Codec>::decode(__input)? {
                    #dec_ord_arms
                    tag => ::std::result::Result::Err(#codec::DecodeError::InvalidTag { what: #what, tag }),
                }
            };
        }
        Data::Union(_) => return compile_error(name.span(), "Codec cannot derive for unions"),
    }

    quote! {
        impl #impl_g #codec::ResidentSize for #name #ty_g #where_c {
            fn resident_size(&self) -> usize {
                #resident
            }
        }
        impl #impl_g #codec::Codec for #name #ty_g #where_c {
            fn encode(&self, __out: &mut impl #codec::BufMut) {
                #encode
            }
            fn decode(__input: &mut &[u8]) -> #codec::Result<Self> {
                #decode
            }
            fn encode_size(&self) -> usize {
                #size
            }
        }
        impl #impl_g #codec::OrderedCodec for #name #ty_g #where_c {
            fn encode_ordered(&self, __out: &mut impl #codec::BufMut) {
                #encode_ordered
            }
            fn decode_ordered(__input: &mut &[u8]) -> #codec::Result<Self> {
                #decode_ordered
            }
            fn encode_ordered_size(&self) -> usize {
                #size_ordered
            }
        }
    }
    .into()
}

/// Mirror a struct of sources, `#[view]` fields and `#[schema]` fields
/// into the faces of a database: `{Name}Write` (the write side, one
/// facade per source and one transaction-bound read facade per view),
/// `{Name}Read` (every field's read facade, bound to one snapshot),
/// `{Name}Builder` (one stream head per source, one `define` slot per
/// view, one nested builder per schema) and `{Name}Handles`. Keyspaces
/// are named after the fields, prefixed by the path of nested schemas
/// (`ledger.users`). Also emits `Name::open(dir)` and `Name::memory()`,
/// returning a `DbBuilder`.
///
/// A `#[schema]` field's type must implement `Declare`; its `declare`
/// runs before this one's.
///
/// The macro never inspects field *types*: each field's faces come from
/// that type's `SourceDecl`, `ViewDecl` or `Schema` impl.
#[proc_macro_derive(Schema, attributes(view, schema, fold3))]
pub fn derive_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let vis = &input.vis;
    let krate = match crate_path(&input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };
    let Data::Struct(s) = &input.data else {
        return compile_error(name.span(), "Schema derives only for structs");
    };
    let Fields::Named(fields) = &s.fields else {
        return compile_error(name.span(), "Schema needs named fields");
    };
    if !input.generics.params.is_empty() {
        return compile_error(name.span(), "Schema structs cannot be generic");
    }

    struct Field<'a> {
        ident: &'a syn::Ident,
        vis: &'a syn::Visibility,
        ty: &'a syn::Type,
        name: String,
    }
    let mut views: Vec<Field> = Vec::new();
    let mut sources: Vec<Field> = Vec::new();
    let mut nested: Vec<Field> = Vec::new();
    for f in &fields.named {
        let ident = f.ident.as_ref().unwrap();
        let field = Field {
            ident,
            vis: &f.vis,
            ty: &f.ty,
            name: ident.to_string(),
        };
        let is_view = f.attrs.iter().any(|a| a.path().is_ident("view"));
        let is_schema = f.attrs.iter().any(|a| a.path().is_ident("schema"));
        match (is_view, is_schema) {
            (true, true) => {
                return compile_error(ident.span(), "a field is either #[view] or #[schema]");
            }
            (true, false) => views.push(field),
            (false, true) => nested.push(field),
            (false, false) => sources.push(field),
        }
    }

    let write = format_ident!("{name}Write");
    let handles = format_ident!("{name}Handles");
    let read = format_ident!("{name}Read");
    let builder = format_ident!("{name}Builder");
    let changes = format_ident!("{name}Changes");
    let feed = format_ident!("{name}Feed");
    let sch = quote!(#krate::schema);

    let s_ident: Vec<_> = sources.iter().map(|f| f.ident).collect();
    let s_vis: Vec<_> = sources.iter().map(|f| f.vis).collect();
    let s_ty: Vec<_> = sources.iter().map(|f| f.ty).collect();
    let s_name: Vec<_> = sources.iter().map(|f| &f.name).collect();
    let v_ident: Vec<_> = views.iter().map(|f| f.ident).collect();
    let v_vis: Vec<_> = views.iter().map(|f| f.vis).collect();
    let v_ty: Vec<_> = views.iter().map(|f| f.ty).collect();
    let v_name: Vec<_> = views.iter().map(|f| &f.name).collect();
    let n_ident: Vec<_> = nested.iter().map(|f| f.ident).collect();
    let n_vis: Vec<_> = nested.iter().map(|f| f.vis).collect();
    let n_ty: Vec<_> = nested.iter().map(|f| f.ty).collect();
    let n_name: Vec<_> = nested.iter().map(|f| &f.name).collect();

    quote! {
        #vis struct #write<'a> {
            #( #s_vis #s_ident: <#s_ty as #sch::SourceDecl>::Write<'a>, )*
            #( #v_vis #v_ident: <#v_ty as #sch::ViewDecl>::Read<'a>, )*
            #( #n_vis #n_ident: <#n_ty as #sch::Schema>::Write<'a>, )*
            #[doc(hidden)]
            __handles: &'a #handles,
            #[doc(hidden)]
            __tx: &'a #sch::TxCell<'a>,
        }

        impl<'a> #write<'a> {
            /// Every source and view as of the transaction's start:
            /// committed state, without this transaction's writes.
            #vis fn read(&self) -> #read<'_> {
                <#name as #sch::Schema>::read(self.__handles, #sch::At::Before(self.__tx))
            }

            /// The epoch this transaction commits as.
            #vis fn epoch(&self) -> u64 {
                self.__tx.epoch()
            }

            /// Write `changes`, as another database of this schema
            /// committed them, to the sources. Write-time checks do
            /// not run; the graph derives the views as for any write.
            #vis fn apply(&mut self, changes: &#changes) -> #krate::Result<()> {
                <#name as #sch::Schema>::apply(self, changes)
            }

            /// Run `f` as a unit within this transaction. If it fails,
            /// or a constraint refuses what it wrote, everything it did
            /// is undone and the error returned, and the transaction
            /// carries on as if `f` never ran.
            #vis fn attempt<T>(
                &mut self,
                f: impl ::std::ops::FnOnce(&mut Self) -> #krate::Result<T>,
            ) -> #krate::Result<T> {
                let tx = self.__tx;
                tx.attempt(|| f(self))
            }
        }

        #vis struct #handles {
            #( #s_ident: <#s_ty as #sch::SourceDecl>::Handle, )*
            #( #v_ident: <#v_ty as #sch::ViewDecl>::Handle, )*
            #( #n_ident: <#n_ty as #sch::Schema>::Handles, )*
        }

        #vis struct #read<'a> {
            #( #s_vis #s_ident: <#s_ty as #sch::SourceDecl>::Read<'a>, )*
            #( #v_vis #v_ident: <#v_ty as #sch::ViewDecl>::Read<'a>, )*
            #( #n_vis #n_ident: <#n_ty as #sch::Schema>::Read<'a>, )*
        }

        /// One commit's changes to every source.
        #[derive(Clone)]
        #vis struct #changes {
            /// The epoch the commit committed as.
            pub epoch: u64,
            #( #s_vis #s_ident: ::std::vec::Vec<<#s_ty as #sch::SourceDecl>::Change>, )*
            #( #n_vis #n_ident: <#n_ty as #sch::Schema>::Changes, )*
        }

        impl #sch::ChangeSet for #changes {
            fn epoch(&self) -> u64 {
                self.epoch
            }

            fn is_empty(&self) -> bool {
                true
                    #( && self.#s_ident.is_empty() )*
                    #( && #sch::ChangeSet::is_empty(&self.#n_ident) )*
            }

            fn invert(self) -> Self {
                #changes {
                    epoch: self.epoch,
                    #( #s_ident: self.#s_ident.into_iter().map(#sch::Invert::invert).collect(), )*
                    #( #n_ident: #sch::ChangeSet::invert(self.#n_ident), )*
                }
            }

            fn encode_into(&self, out: &mut ::std::vec::Vec<u8>) {
                #krate::store::codec::Codec::encode(&self.epoch, out);
                #( #krate::store::codec::Codec::encode(&self.#s_ident, out); )*
                #( #sch::ChangeSet::encode_into(&self.#n_ident, out); )*
            }

            fn decode_from(b: &mut &[u8]) -> #krate::store::codec::Result<Self> {
                ::std::result::Result::Ok(#changes {
                    epoch: #krate::store::codec::Codec::decode(b)?,
                    #( #s_ident: #krate::store::codec::Codec::decode(b)?, )*
                    #( #n_ident: #sch::ChangeSet::decode_from(b)?, )*
                })
            }
        }

        #[doc(hidden)]
        #vis struct #feed {
            #( #s_ident: #sch::Tap<<#s_ty as #sch::SourceDecl>::Change>, )*
            #( #n_ident: <#n_ty as #sch::Schema>::Feed, )*
        }

        impl #sch::CommitFeed for #feed {
            type Changes = #changes;

            fn epoch(&mut self) -> ::std::option::Option<u64> {
                #sch::first_epoch([
                    #( self.#s_ident.epoch(), )*
                    #( #sch::CommitFeed::epoch(&mut self.#n_ident), )*
                ])
            }

            fn take(&mut self, epoch: u64) -> #changes {
                #changes {
                    epoch,
                    #( #s_ident: self.#s_ident.take(epoch), )*
                    #( #n_ident: #sch::CommitFeed::take(&mut self.#n_ident, epoch), )*
                }
            }
        }

        #vis struct #builder {
            #( #s_vis #s_ident: <#s_ty as #sch::SourceDecl>::Head, )*
            #( #v_vis #v_ident: <#v_ty as #sch::ViewDecl>::Slot, )*
            #( #n_vis #n_ident: <#n_ty as #sch::Schema>::Builder, )*
        }

        impl #sch::Schema for #name {
            type Handles = #handles;
            type Builder = #builder;
            type Write<'a> = #write<'a>;
            type Read<'a> = #read<'a>;
            type Changes = #changes;
            type Feed = #feed;

            fn feed(read: &#read<'_>) -> #feed {
                #feed {
                    #( #s_ident: #sch::Tap::new(<#s_ty as #sch::SourceDecl>::subscribe(&read.#s_ident)), )*
                    #( #n_ident: <#n_ty as #sch::Schema>::feed(&read.#n_ident), )*
                }
            }

            fn attempt<'a, T>(
                write: &mut #write<'a>,
                f: impl ::std::ops::FnOnce(&mut #write<'a>) -> #krate::Result<T>,
            ) -> #krate::Result<T> {
                write.attempt(f)
            }

            fn apply(write: &#write<'_>, changes: &#changes) -> #krate::Result<()> {
                #( <#s_ty as #sch::SourceDecl>::apply(&write.#s_ident, &changes.#s_ident)?; )*
                #( <#n_ty as #sch::Schema>::apply(&write.#n_ident, &changes.#n_ident)?; )*
                ::std::result::Result::Ok(())
            }

            fn open(prefix: &str, build: &#sch::Shared) -> #builder {
                #builder {
                    #( #s_ident: <#s_ty as #sch::SourceDecl>::open(
                        ::std::sync::Arc::from(format!("{prefix}{}", #s_name)), build), )*
                    #( #v_ident: <#v_ty as #sch::ViewDecl>::open(
                        ::std::sync::Arc::from(format!("{prefix}{}", #v_name)), build), )*
                    #( #n_ident: <#n_ty as #sch::Schema>::open(
                        &format!("{prefix}{}.", #n_name), build), )*
                }
            }

            fn declare_nested(b: &mut #builder) {
                #(
                    <#n_ty as #sch::Schema>::declare_nested(&mut b.#n_ident);
                    <#n_ty as #sch::Declare>::declare(&mut b.#n_ident);
                )*
            }

            fn finish(b: #builder) -> #handles {
                #handles {
                    #( #s_ident: <#s_ty as #sch::SourceDecl>::finish(b.#s_ident), )*
                    #( #v_ident: <#v_ty as #sch::ViewDecl>::finish(b.#v_ident), )*
                    #( #n_ident: <#n_ty as #sch::Schema>::finish(b.#n_ident), )*
                }
            }

            fn write<'a>(h: &'a #handles, tx: &'a #sch::TxCell<'a>) -> #write<'a> {
                #write {
                    #( #s_ident: <#s_ty as #sch::SourceDecl>::write(&h.#s_ident, tx), )*
                    #( #v_ident: <#v_ty as #sch::ViewDecl>::read(&h.#v_ident, #sch::At::During(tx)), )*
                    #( #n_ident: <#n_ty as #sch::Schema>::write(&h.#n_ident, tx), )*
                    __handles: h,
                    __tx: tx,
                }
            }

            fn read<'a>(h: &'a #handles, at: #sch::At<'a>) -> #read<'a> {
                #read {
                    #( #s_ident: <#s_ty as #sch::SourceDecl>::read(&h.#s_ident, at), )*
                    #( #v_ident: <#v_ty as #sch::ViewDecl>::read(&h.#v_ident, at), )*
                    #( #n_ident: <#n_ty as #sch::Schema>::read(&h.#n_ident, at), )*
                }
            }
        }

        impl #sch::Fields for #name {
            fn __touch(&self) {
                #( let _ = &self.#s_ident; )*
                #( let _ = &self.#v_ident; )*
                #( let _ = &self.#n_ident; )*
            }
        }

        impl #name {
            /// A durable database of this schema at `dir`.
            #vis fn open(dir: impl AsRef<::std::path::Path>) -> #sch::DbBuilder<Self>
            where
                Self: #sch::Declare,
            {
                #sch::DbBuilder::open(dir)
            }

            /// A memory-only database of this schema.
            #vis fn memory() -> #sch::DbBuilder<Self>
            where
                Self: #sch::Declare,
            {
                #sch::DbBuilder::memory()
            }

            /// A database of this schema over a store configured by hand.
            #vis fn with(store: #krate::StoreBuilder) -> #sch::DbBuilder<Self>
            where
                Self: #sch::Declare,
            {
                #sch::DbBuilder::with(store)
            }
        }
    }
    .into()
}
