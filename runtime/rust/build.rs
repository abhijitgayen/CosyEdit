fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .compile(&["../python/grpc/cosyvoice.proto"], &["../python/grpc"])?;
    Ok(())
}
