import Testing
@testable import SwiftMatrix

struct TensorStridedIteratorTests {
    @Test func transposedRank2() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]]).transposed()
        var iterator = t.makeStridedIterator()
        var elements = [Int]()
        while let e = iterator.next() {
            elements.append(e)
        }
        #expect(elements == [1, 4, 2, 5, 3, 6])
    }

    @Test func permutedRank3() {
        let t = Tensor(shape: [2, 3, 4], elements: Array(1...24))
        let p = t.permuted(axes: [2, 0, 1])
        var iterator = p.makeStridedIterator()
        var elements = [Int]()
        while let e = iterator.next() {
            elements.append(e)
        }
        #expect(elements == Array(p))
    }

    @Test func slicedTensor() {
        let t = Tensor(shape: [4, 3], elements: Array(1...12))
        let s = t.slice(axis: 0, range: 1..<3)
        var iterator = s.makeStridedIterator()
        var elements = [Int]()
        while let e = iterator.next() {
            elements.append(e)
        }
        #expect(elements == [4, 5, 6, 7, 8, 9])
    }

    @Test func rank1() {
        let t = Tensor(shape: [5], elements: [10, 20, 30, 40, 50])
        var iterator = t.makeStridedIterator()
        var elements = [Int]()
        while let e = iterator.next() {
            elements.append(e)
        }
        #expect(elements == [10, 20, 30, 40, 50])
    }

    @Test func scalar() {
        let t = Tensor(shape: [], elements: [42])
        var iterator = t.makeStridedIterator()
        #expect(iterator.next() == 42)
        #expect(iterator.next() == nil)
    }

    @Test func emptyTensor() {
        let t = Tensor(shape: [0, 3], repeating: 0)
        var iterator = t.makeStridedIterator()
        #expect(iterator.next() == nil)
    }

    @Test func matchesArrayConversion() {
        let t = Tensor(shape: [2, 3, 2], elements: Array(1...12))
        let p = t.permuted(axes: [1, 2, 0])
        var iterator = p.makeStridedIterator()
        var iterElements = [Int]()
        while let e = iterator.next() {
            iterElements.append(e)
        }
        #expect(iterElements == Array(p))
    }

    @Test func contiguousElements() {
        // Verify that contiguousElements uses StridedIterator for non-contiguous tensors
        let t = Tensor([[1, 2, 3], [4, 5, 6]]).transposed()
        let elements = t.contiguousElements()
        #expect(elements == [1, 4, 2, 5, 3, 6])
    }
}
