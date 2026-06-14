FROM rust:1.88-bookworm AS builder

WORKDIR /app

COPY backend/Cargo.toml backend/Cargo.lock ./backend/
COPY backend/src ./backend/src

WORKDIR /app/backend
RUN cargo build --release

FROM debian:bookworm-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/backend/target/release/backend ./backend
COPY model ./model

ENV PORT=8080

CMD ["./backend"]