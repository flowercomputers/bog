# bog

bog is a library for incremental dataflow over an (ACID) store. It its currently available in beta.

## repo layout

- at the root of the crate is the core dataflow/storage engine
- `derive/` provides derive macros for more easily constructing valid bog programs
- `examples/` includes simple OLTP examples for both derived and procedurally constructed bog programs
