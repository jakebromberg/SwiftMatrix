import Testing
@testable import SwiftMatrix

struct TensorPermuteTests {
    @Test func permuteRank2() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]])
        let p = t.permuted(axes: [1, 0])
        #expect(p.shape == [3, 2])
        #expect(Array(p) == [1, 4, 2, 5, 3, 6])
    }

    @Test func permuteRank3() {
        let t = Tensor(shape: [2, 3, 4], elements: Array(1...24))
        let p = t.permuted(axes: [2, 0, 1])
        #expect(p.shape == [4, 2, 3])
        // Original t[0,0,0]=1, t[0,0,1]=2, ...
        // Permuted: new axis 0 = old axis 2, new axis 1 = old axis 0, new axis 2 = old axis 1
        // p[0,0,0] = t[0,0,0] = 1
        // p[0,0,1] = t[0,1,0] = 5
        // p[0,0,2] = t[0,2,0] = 9
        // p[0,1,0] = t[1,0,0] = 13
        #expect(p[0, 0, 0] == 1)
        #expect(p[0, 0, 1] == 5)
        #expect(p[0, 0, 2] == 9)
        #expect(p[0, 1, 0] == 13)
    }

    @Test func permuteIdentity() {
        let t = Tensor([[1, 2], [3, 4]])
        let p = t.permuted(axes: [0, 1])
        #expect(p.shape == [2, 2])
        #expect(p.isContiguous)
        #expect(Array(p) == [1, 2, 3, 4])
    }

    @Test func transposedRank2() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]])
        let tr = t.transposed()
        #expect(tr.shape == [3, 2])
        #expect(Array(tr) == [1, 4, 2, 5, 3, 6])
    }
}

struct TensorReshapeTests {
    @Test(arguments: [
        (from: [2, 3], to: [6]),
        (from: [6],    to: [2, 3]),
        (from: [2, 3], to: [3, 2]),
        (from: [1, 6], to: [6, 1]),
    ])
    func reshapePreservesElements(from: [Int], to: [Int]) {
        let elements = [1, 2, 3, 4, 5, 6]
        let t = Tensor(shape: from, elements: elements)
        let r = t.reshaped(to: to)
        #expect(r.shape == to)
        #expect(Array(r) == elements)
    }
}

struct TensorSliceTests {
    @Test func sliceFirstAxis() {
        let t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        let s = t.slice(axis: 0, range: 1..<3)
        #expect(s.shape == [2, 3])
        #expect(Array(s) == [4, 5, 6, 7, 8, 9])
    }

    @Test func sliceSecondAxis() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]])
        let s = t.slice(axis: 1, range: 0..<2)
        #expect(s.shape == [2, 2])
        #expect(Array(s) == [1, 2, 4, 5])
    }

    @Test func sliceSingleRow() {
        let t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        let s = t.slice(axis: 0, range: 1..<2)
        #expect(s.shape == [1, 3])
        #expect(Array(s) == [4, 5, 6])
    }

    @Test func slicedThenArray() {
        let t = Tensor(shape: [4, 3], elements: Array(1...12))
        let s = t.slice(axis: 0, range: 1..<3)
        #expect(Array(s) == [4, 5, 6, 7, 8, 9])
    }
}
