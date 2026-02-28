import Foundation

/// Prevents dead-code elimination in release builds by consuming a value.
@inline(never)
func blackHole<T>(_ value: T) {
    withExtendedLifetime(value) {}
}

/// Result of a single benchmark run across all three libraries.
struct BenchmarkResult {
    let operation: String
    let elementType: String
    let size: String
    let swiftMatrix: Duration
    let surge: Duration
    let matft: Duration
}

/// Measures the median duration of `body` over `iterations` runs, preceded by one warmup.
///
/// - Parameters:
///   - iterations: Number of timed iterations to run.
///   - body: The closure to benchmark.
/// - Returns: The median duration across all timed iterations.
func measure(iterations: Int, _ body: () -> Void) -> Duration {
    // Warmup
    body()

    let clock = ContinuousClock()
    var durations: [Duration] = []
    durations.reserveCapacity(iterations)

    for _ in 0..<iterations {
        let elapsed = clock.measure { body() }
        durations.append(elapsed)
    }

    durations.sort()
    return durations[durations.count / 2]
}

/// Returns the number of iterations for the given matrix size.
func iterations(forSize n: Int) -> Int {
    switch n {
    case ...64: return 50
    case ...256: return 20
    default: return 10
    }
}

/// Formats a `Duration` as milliseconds with 3 decimal places.
func formatMs(_ duration: Duration) -> String {
    let ns = Double(duration.components.seconds) * 1_000_000_000
        + Double(duration.components.attoseconds) / 1_000_000_000
    let ms = ns / 1_000_000
    return String(format: "%.3f ms", ms)
}

/// Pads a string to the given width with trailing spaces.
private func pad(_ s: String, to width: Int) -> String {
    s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
}

/// Prints a formatted benchmark comparison table.
///
/// - Parameter results: Array of `BenchmarkResult` values for a single operation/type combo.
func printTable(title: String, results: [BenchmarkResult]) {
    let divider = String(repeating: "-", count: 62)

    print()
    print(title)
    print(divider)
    print(" \(pad("Size", to: 11))| \(pad("SwiftMatrix", to: 12))| \(pad("Surge", to: 12))| \(pad("Matft", to: 12))")
    print(divider)
    for r in results {
        print(" \(pad(r.size, to: 11))| \(pad(formatMs(r.swiftMatrix), to: 12))| \(pad(formatMs(r.surge), to: 12))| \(pad(formatMs(r.matft), to: 12))")
    }
    print(divider)
}
