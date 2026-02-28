import Foundation

// MARK: - Configuration

let sizes = [64, 256, 1024]

// MARK: - Data generation

/// Generates a flat array of random values in [0, 1).
func generateFloatData(count: Int) -> [Float] {
    (0..<count).map { _ in Float.random(in: 0..<1) }
}

func generateDoubleData(count: Int) -> [Double] {
    (0..<count).map { _ in Double.random(in: 0..<1) }
}

// MARK: - Benchmark runner

typealias FloatBenchmark = (_ a: [Float], _ b: [Float], _ n: Int) -> Void
typealias FloatUnaryBenchmark = (_ a: [Float], _ n: Int) -> Void
typealias DoubleBenchmark = (_ a: [Double], _ b: [Double], _ n: Int) -> Void
typealias DoubleUnaryBenchmark = (_ a: [Double], _ n: Int) -> Void

func runFloatBenchmarks(
    operation: String,
    swiftMatrix: @escaping FloatBenchmark,
    surge: @escaping FloatBenchmark,
    matft: @escaping FloatBenchmark
) -> [BenchmarkResult] {
    sizes.map { n in
        let count = n * n
        let a = generateFloatData(count: count)
        let b = generateFloatData(count: count)
        let iters = iterations(forSize: n)

        let sm = measure(iterations: iters) { swiftMatrix(a, b, n) }
        let su = measure(iterations: iters) { surge(a, b, n) }
        let mf = measure(iterations: iters) { matft(a, b, n) }

        return BenchmarkResult(
            operation: operation, elementType: "Float", size: "\(n)x\(n)",
            swiftMatrix: sm, surge: su, matft: mf)
    }
}

func runFloatUnaryBenchmarks(
    operation: String,
    swiftMatrix: @escaping FloatUnaryBenchmark,
    surge: @escaping FloatUnaryBenchmark,
    matft: @escaping FloatUnaryBenchmark
) -> [BenchmarkResult] {
    sizes.map { n in
        let count = n * n
        let a = generateFloatData(count: count)
        let iters = iterations(forSize: n)

        let sm = measure(iterations: iters) { swiftMatrix(a, n) }
        let su = measure(iterations: iters) { surge(a, n) }
        let mf = measure(iterations: iters) { matft(a, n) }

        return BenchmarkResult(
            operation: operation, elementType: "Float", size: "\(n)x\(n)",
            swiftMatrix: sm, surge: su, matft: mf)
    }
}

func runDoubleBenchmarks(
    operation: String,
    swiftMatrix: @escaping DoubleBenchmark,
    surge: @escaping DoubleBenchmark,
    matft: @escaping DoubleBenchmark
) -> [BenchmarkResult] {
    sizes.map { n in
        let count = n * n
        let a = generateDoubleData(count: count)
        let b = generateDoubleData(count: count)
        let iters = iterations(forSize: n)

        let sm = measure(iterations: iters) { swiftMatrix(a, b, n) }
        let su = measure(iterations: iters) { surge(a, b, n) }
        let mf = measure(iterations: iters) { matft(a, b, n) }

        return BenchmarkResult(
            operation: operation, elementType: "Double", size: "\(n)x\(n)",
            swiftMatrix: sm, surge: su, matft: mf)
    }
}

func runDoubleUnaryBenchmarks(
    operation: String,
    swiftMatrix: @escaping DoubleUnaryBenchmark,
    surge: @escaping DoubleUnaryBenchmark,
    matft: @escaping DoubleUnaryBenchmark
) -> [BenchmarkResult] {
    sizes.map { n in
        let count = n * n
        let a = generateDoubleData(count: count)
        let iters = iterations(forSize: n)

        let sm = measure(iterations: iters) { swiftMatrix(a, n) }
        let su = measure(iterations: iters) { surge(a, n) }
        let mf = measure(iterations: iters) { matft(a, n) }

        return BenchmarkResult(
            operation: operation, elementType: "Double", size: "\(n)x\(n)",
            swiftMatrix: sm, surge: su, matft: mf)
    }
}

// MARK: - Main

print("SwiftMatrix Performance Benchmarks")
print("===================================")

// Matrix Addition
printTable(
    title: "Matrix Addition (Float)",
    results: runFloatBenchmarks(
        operation: "Matrix Addition",
        swiftMatrix: SwiftMatrixBenchmarks.addFloat,
        surge: SurgeBenchmarks.addFloat,
        matft: MatftBenchmarks.addFloat))

printTable(
    title: "Matrix Addition (Double)",
    results: runDoubleBenchmarks(
        operation: "Matrix Addition",
        swiftMatrix: SwiftMatrixBenchmarks.addDouble,
        surge: SurgeBenchmarks.addDouble,
        matft: MatftBenchmarks.addDouble))

// Matrix Multiplication
printTable(
    title: "Matrix Multiplication (Float)",
    results: runFloatBenchmarks(
        operation: "Matrix Multiplication",
        swiftMatrix: SwiftMatrixBenchmarks.matmulFloat,
        surge: SurgeBenchmarks.matmulFloat,
        matft: MatftBenchmarks.matmulFloat))

printTable(
    title: "Matrix Multiplication (Double)",
    results: runDoubleBenchmarks(
        operation: "Matrix Multiplication",
        swiftMatrix: SwiftMatrixBenchmarks.matmulDouble,
        surge: SurgeBenchmarks.matmulDouble,
        matft: MatftBenchmarks.matmulDouble))

// Sum Reduction
printTable(
    title: "Sum Reduction (Float)",
    results: runFloatUnaryBenchmarks(
        operation: "Sum Reduction",
        swiftMatrix: SwiftMatrixBenchmarks.sumFloat,
        surge: SurgeBenchmarks.sumFloat,
        matft: MatftBenchmarks.sumFloat))

printTable(
    title: "Sum Reduction (Double)",
    results: runDoubleUnaryBenchmarks(
        operation: "Sum Reduction",
        swiftMatrix: SwiftMatrixBenchmarks.sumDouble,
        surge: SurgeBenchmarks.sumDouble,
        matft: MatftBenchmarks.sumDouble))

// Dot Product
printTable(
    title: "Dot Product (Float)",
    results: runFloatBenchmarks(
        operation: "Dot Product",
        swiftMatrix: SwiftMatrixBenchmarks.dotFloat,
        surge: SurgeBenchmarks.dotFloat,
        matft: MatftBenchmarks.dotFloat))

printTable(
    title: "Dot Product (Double)",
    results: runDoubleBenchmarks(
        operation: "Dot Product",
        swiftMatrix: SwiftMatrixBenchmarks.dotDouble,
        surge: SurgeBenchmarks.dotDouble,
        matft: MatftBenchmarks.dotDouble))
