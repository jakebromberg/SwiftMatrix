import Testing
@testable import SwiftMatrix

struct TensorLazyEvaluateTests {
    @Test func lazyAddMatchesEager() {
        let a = Tensor(shape: [3], elements: [1, 2, 3])
        let b = Tensor(shape: [3], elements: [4, 5, 6])
        let result = Tensor(evaluating: a.lazy + b.lazy)
        #expect(result == a + b)
    }

    @Test func lazySubMatchesEager() {
        let a = Tensor(shape: [3], elements: [10, 20, 30])
        let b = Tensor(shape: [3], elements: [1, 2, 3])
        let result = Tensor(evaluating: a.lazy - b.lazy)
        #expect(result == a - b)
    }

    @Test func lazyMulMatchesEager() {
        let a = Tensor(shape: [3], elements: [2, 3, 4])
        let b = Tensor(shape: [3], elements: [5, 6, 7])
        let result = Tensor(evaluating: a.lazy * b.lazy)
        #expect(result == a * b)
    }

    @Test func lazyDivMatchesEager() {
        let a = Tensor(shape: [2], elements: [10.0, 20.0])
        let b = Tensor(shape: [2], elements: [2.0, 5.0])
        let result = Tensor(evaluating: a.lazy / b.lazy)
        #expect(result == a / b)
    }
}

struct TensorLazyFusionTests {
    @Test func fusedAddMul() {
        // a + b * c in one pass, no intermediate allocations
        let a = Tensor(shape: [3], elements: [1, 2, 3])
        let b = Tensor(shape: [3], elements: [4, 5, 6])
        let c = Tensor(shape: [3], elements: [2, 2, 2])
        let result = Tensor(evaluating: a.lazy + b.lazy * c.lazy)
        // b*c = [8,10,12], a + b*c = [9,12,15]
        #expect(result == Tensor(shape: [3], elements: [9, 12, 15]))
    }

    @Test func fusedSubDiv() {
        let a = Tensor(shape: [2], elements: [10.0, 20.0])
        let b = Tensor(shape: [2], elements: [2.0, 4.0])
        let c = Tensor(shape: [2], elements: [1.0, 2.0])
        let result = Tensor(evaluating: a.lazy / b.lazy - c.lazy)
        // a/b = [5,5], - c = [4,3]
        #expect(result == Tensor(shape: [2], elements: [4.0, 3.0]))
    }

    @Test func tripleAdd() {
        let a = Tensor(shape: [2], elements: [1, 2])
        let b = Tensor(shape: [2], elements: [3, 4])
        let c = Tensor(shape: [2], elements: [5, 6])
        let result = Tensor(evaluating: a.lazy + b.lazy + c.lazy)
        #expect(result == Tensor(shape: [2], elements: [9, 12]))
    }
}

struct TensorLazyNegationTests {
    @Test func lazyNegation() {
        let a = Tensor(shape: [3], elements: [1, 2, 3])
        let result = Tensor(evaluating: -a.lazy)
        #expect(result == -a)
    }

    @Test func negatedInExpression() {
        let a = Tensor(shape: [3], elements: [1, 2, 3])
        let b = Tensor(shape: [3], elements: [4, 5, 6])
        let result = Tensor(evaluating: b.lazy + (-a.lazy))
        #expect(result == b - a)
    }
}

struct TensorLazyNonContiguousTests {
    @Test func nonContiguousLeaves() {
        let a = Tensor([[1, 2, 3], [4, 5, 6]]).transposed() // [3,2]
        let b = Tensor(shape: [3, 2], elements: [10, 10, 10, 10, 10, 10])
        let result = Tensor(evaluating: a.lazy + b.lazy)
        #expect(result == a + b)
    }
}

struct TensorLazyRank2Tests {
    @Test func rank2LazyAdd() {
        let a = Tensor([[1, 2], [3, 4]])
        let b = Tensor([[10, 20], [30, 40]])
        let result = Tensor(evaluating: a.lazy + b.lazy)
        #expect(result == Tensor([[11, 22], [33, 44]]))
    }
}

struct TensorEagerUnaffectedTests {
    @Test func eagerAddStillWorks() {
        // Verify a + b (no .lazy) still uses eager operators
        let a = Tensor(shape: [3], elements: [1, 2, 3])
        let b = Tensor(shape: [3], elements: [4, 5, 6])
        let result = a + b
        #expect(result == Tensor(shape: [3], elements: [5, 7, 9]))
    }

    @Test func eagerMulStillWorks() {
        let a = Tensor(shape: [2], elements: [2, 3])
        let b = Tensor(shape: [2], elements: [4, 5])
        let result = a * b
        #expect(result == Tensor(shape: [2], elements: [8, 15]))
    }
}
