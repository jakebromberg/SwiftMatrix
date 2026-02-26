import Testing
@testable import SwiftMatrix

struct TensorContiguousBufferTests {
    @Test func contiguousTensorProvidesBuffer() {
        let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        let result = t.withContiguousStorageIfAvailable { buffer -> [Int] in
            Array(buffer)
        }
        #expect(result == [1, 2, 3, 4, 5, 6])
    }

    @Test func nonContiguousTransposedReturnsNil() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]]).transposed()
        let result = t.withContiguousStorageIfAvailable { buffer -> [Int] in
            Array(buffer)
        }
        #expect(result == nil)
    }

    @Test func nonContiguousSlicedReturnsNil() {
        let t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        let sliced = t.slice(axis: 0, range: 1..<3)
        let result = sliced.withContiguousStorageIfAvailable { buffer -> [Int] in
            Array(buffer)
        }
        #expect(result == nil)
    }

    @Test func stdlibReduceViaBuffer() {
        let t = Tensor(shape: [3], elements: [10, 20, 30])
        let sum = t.withContiguousStorageIfAvailable { buffer -> Int in
            buffer.reduce(0, +)
        }
        #expect(sum == 60)
    }

    @Test func stdlibElementsEqualViaBuffer() {
        let a = Tensor(shape: [4], elements: [1, 2, 3, 4])
        let b = Tensor(shape: [4], elements: [1, 2, 3, 4])
        let equal = a.withContiguousStorageIfAvailable { bufA in
            b.withContiguousStorageIfAvailable { bufB in
                bufA.elementsEqual(bufB)
            }
        }
        #expect(equal == Optional(Optional(true)))
    }

    @Test func scalarTensorProvidesBuffer() {
        let t = Tensor(shape: [], elements: [42])
        let result = t.withContiguousStorageIfAvailable { buffer -> Int in
            buffer[0]
        }
        #expect(result == 42)
    }

    @Test func rank1TensorProvidesBuffer() {
        let t = Tensor(shape: [5], elements: [10, 20, 30, 40, 50])
        let result = t.withContiguousStorageIfAvailable { buffer -> Int in
            buffer.count
        }
        #expect(result == 5)
    }
}

struct TensorCachedLogicalStridesTests {
    @Test func storageIndexNonContiguousUsesCache() {
        // A transposed 2x3 -> 3x2 tensor
        let t = Tensor(
            storage: [1, 2, 3, 4, 5, 6],
            shape: [3, 2],
            strides: [1, 3],
            offset: 0,
            isContiguous: false
        )
        // Verify iteration still produces the correct elements
        #expect(Array(t) == [1, 4, 2, 5, 3, 6])
    }

    @Test func logicalStridesMatchRowMajor() {
        let t = Tensor(shape: [2, 3, 4], repeating: 0)
        #expect(t.logicalStrides == [12, 4, 1])
    }

    @Test func logicalStridesForViews() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]]).transposed()
        // Transposed shape is [3, 2], so logical strides should be [2, 1]
        #expect(t.logicalStrides == [2, 1])
        // Physical strides are [1, 3] (transposed)
        #expect(t.strides == [1, 3])
    }

    @Test func logicalStridesForSlice() {
        let t = Tensor(shape: [4, 3], elements: Array(1...12))
        let s = t.slice(axis: 0, range: 1..<3)
        #expect(s.logicalStrides == [3, 1])
        #expect(s.strides == [3, 1])
    }

    @Test func logicalStridesScalar() {
        let t = Tensor(shape: [], elements: [42])
        #expect(t.logicalStrides == [])
    }

    @Test func contiguousTensorSharesStridesAndLogicalStrides() {
        let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        #expect(t.logicalStrides == t.strides)
    }
}
