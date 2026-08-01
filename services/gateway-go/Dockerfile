FROM golang:1.22-alpine AS build
# Note: docker-compose.dev.yml targets this "build" stage directly (has
# the Go toolchain) instead of the final runtime stage below, which only
# contains the compiled binary and cannot run `go run`.
WORKDIR /src
COPY go.mod ./
COPY . .
RUN go build -o /out/gateway ./cmd/gateway

FROM alpine:3.20
COPY --from=build /out/gateway /usr/local/bin/gateway
EXPOSE 8080
ENTRYPOINT ["gateway"]
