use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn make_input(len: usize) -> Vec<u8> {
    (0..len).map(|i| i as u8).collect()
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    for &size in &[16usize, 64, 256, 1024, 16 * 1024] {
        let input = make_input(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("base85", size), &input, |b, input| {
            b.iter(|| base85::encode(black_box(input)))
        });
        group.bench_with_input(BenchmarkId::new("base85_simd", size), &input, |b, input| {
            b.iter(|| base85_simd::encode(black_box(input)))
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    for &size in &[16usize, 64, 256, 1024, 16 * 1024] {
        let input = make_input(size);
        let encoded = base85::encode(&input);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("base85", size), &encoded, |b, e| {
            b.iter(|| base85::decode(black_box(e)).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("base85_simd", size), &encoded, |b, e| {
            b.iter(|| base85_simd::decode(black_box(e)).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
