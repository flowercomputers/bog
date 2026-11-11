//! Constraints: everything that can refuse a transaction.
//!
//! - On a source head, `reject`/`check` run at write time, in the write
//!   method; `foreign_key` is dataflow; `unique_key` is an index the
//!   table maintains and checks at write time.
//! - On a view slot, `reject`/`check` run after the graph, over the rows
//!   the transaction left in the view.
//! - [`unique`](crate::stream::Grouped::unique) enforces one row per key as an
//!   operator.
//!
//! Every refusal is an [`Error::Rejected`]; the
//! predicate forms carry a [`Rejected`].

use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use crate::{
    Change, Delta, Mergeless, Nary, Register,
    database::DbTx,
    error::{Error, Result},
    ops::{Reject, Take, dirty_keys_into, read_group},
    state::{Declare as DeclareKeyspace, State, Stored, When},
    tx::{KeyspaceId, WriteTx},
};

use crate::schema::{
    Data, GroupedFeed, Keyed, Rejected,
    source::{CellHead, EventsHead, MultisetHead, SetHead, TableHead},
    view::{GroupSlot, MapSlot, OrderedSlot, SlotCore, ValueSlot},
};

type Shared = Arc<dyn std::error::Error + Send + Sync>;

mod source;
mod unique;
mod unique_key;
mod view;

pub(crate) use unique::Unique;
use unique_key::Alt;
pub(crate) use unique_key::AltKey;
