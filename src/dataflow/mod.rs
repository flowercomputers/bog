//! The dataflow runtime: the buffers that flow along a graph's edges,
//! what an operator is, the graph itself ([`OpDag`]) and the [state]
//! handles through which a node's children read its output. No operators
//! live here; those are in [`ops`](crate::ops).

mod buffer;
pub use buffer::{Bag, Buffer, Delta, Register, vec_scratch};

mod node;
pub use node::{Binary, Nary, Node, Unary};

mod fused;
pub use fused::{Emit, FusedLinears, Ident, Then};

pub mod graph;
pub use graph::{DEFAULT_GRAPH_RETAIN, Fingerprint, NodeRef, OpDag};

pub mod state;

mod build;
pub use build::{Build, Shared};
