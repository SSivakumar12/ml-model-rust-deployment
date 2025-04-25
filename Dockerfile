# Stage 1: Plan dependencies
FROM rust:latest as planner
WORKDIR /app
COPY . .
RUN cargo install cargo-chef
RUN cargo chef prepare --recipe-path recipe.json

# Stage 2: Cache dependencies
FROM rust:latest as cacher
WORKDIR /app
RUN cargo install cargo-chef
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

# Stage 3: Build application
FROM rust:latest as builder
WORKDIR /app
COPY . .
COPY --from=cacher /app/target /app/target
COPY --from=cacher /usr/local/cargo /usr/local/cargo
RUN cargo build --release

# Stage 4: Runtime - using a distroless image to reduce attack risk
FROM gcr.io/distroless/cc as runtime
COPY --from=builder /app/target/release/ml-model-rust-deployment /app/ml-model-rust-deployment
COPY --from=builder /app/model_weights model_weights
CMD ["/app/ml-model-rust-deployment"]
