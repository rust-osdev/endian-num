FROM rust:latest
ENV CARGO_INCREMENTAL=0 \
    CARGO_TERM_COLOR=always

WORKDIR /root/endian-num
COPY . .
RUN set -ex; \
    cargo test --all-features; \
    cargo clean
