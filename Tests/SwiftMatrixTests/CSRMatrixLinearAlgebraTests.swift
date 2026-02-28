import Testing
@testable import SwiftMatrix

struct CSRMatrixMatvecTests {
    @Test func basic() {
        // [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
        let a = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 2, 3, 5],
            columnIndices: [0, 2, 1, 0, 2],
            values: [1, 2, 3, 4, 5]
        )
        let x = Tensor(shape: [3], elements: [1, 2, 3])
        // row 0: 1*1 + 2*3 = 7
        // row 1: 3*2 = 6
        // row 2: 4*1 + 5*3 = 19
        let result = CSRMatrix.matvec(a, x)
        #expect(result.shape == [3])
        #expect(Array(result) == [7, 6, 19])
    }

    @Test func identityMatrix() {
        let eye = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 1, 2, 3],
            columnIndices: [0, 1, 2],
            values: [1, 1, 1]
        )
        let x = Tensor(shape: [3], elements: [10, 20, 30])
        let result = CSRMatrix.matvec(eye, x)
        #expect(Array(result) == [10, 20, 30])
    }

    @Test func emptyMatrix() {
        let a = CSRMatrix<Int>(rows: 2, columns: 3)
        let x = Tensor(shape: [3], elements: [1, 2, 3])
        let result = CSRMatrix.matvec(a, x)
        #expect(Array(result) == [0, 0])
    }

    @Test func emptyRows() {
        // Row 1 is empty: [[1, 2], [0, 0], [3, 0]]
        let a = CSRMatrix(
            rows: 3, columns: 2,
            rowPointers: [0, 2, 2, 3],
            columnIndices: [0, 1, 0],
            values: [1, 2, 3]
        )
        let x = Tensor(shape: [2], elements: [10, 20])
        let result = CSRMatrix.matvec(a, x)
        #expect(Array(result) == [50, 0, 30])
    }

    @Test func singleRow() {
        let a = CSRMatrix(
            rows: 1, columns: 3,
            rowPointers: [0, 2],
            columnIndices: [0, 2],
            values: [3, 5]
        )
        let x = Tensor(shape: [3], elements: [1, 2, 3])
        // 3*1 + 5*3 = 18
        let result = CSRMatrix.matvec(a, x)
        #expect(Array(result) == [18])
    }

    @Test func matchesDense() {
        let a = CSRMatrix(
            rows: 2, columns: 3,
            rowPointers: [0, 2, 3],
            columnIndices: [0, 2, 1],
            values: [1, 2, 3]
        )
        let x = Tensor(shape: [3], elements: [4, 5, 6])
        let sparse = CSRMatrix.matvec(a, x)
        // Dense matmul: reshape x to [3,1], matmul, flatten
        let dense = Tensor.matmul(
            a.toTensor(),
            Tensor(shape: [3, 1], elements: Array(x))
        )
        for i in 0..<sparse.count {
            #expect(sparse[i] == dense[i, 0])
        }
    }
}

struct CSRMatrixSparseDenseMatmulTests {
    @Test func basic() {
        // A: [[1, 0], [0, 2]]  B: [[3, 4], [5, 6]]
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1, 2]
        )
        let b = Tensor([[3, 4], [5, 6]])
        // row 0: 1*[3,4] = [3, 4]
        // row 1: 2*[5,6] = [10, 12]
        let result = CSRMatrix.matmul(a, b)
        #expect(result.shape == [2, 2])
        #expect(Array(result) == [3, 4, 10, 12])
    }

    @Test func identityMatrix() {
        let eye = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1, 1]
        )
        let b = Tensor([[10, 20, 30], [40, 50, 60]])
        let result = CSRMatrix.matmul(eye, b)
        #expect(Array(result) == Array(b))
    }

    @Test func emptyMatrix() {
        let a = CSRMatrix<Int>(rows: 2, columns: 3)
        let b = Tensor([[1, 2], [3, 4], [5, 6]])
        let result = CSRMatrix.matmul(a, b)
        #expect(result.shape == [2, 2])
        #expect(Array(result) == [0, 0, 0, 0])
    }

    @Test func emptyRows() {
        // Row 1 is empty
        let a = CSRMatrix(
            rows: 3, columns: 2,
            rowPointers: [0, 1, 1, 2],
            columnIndices: [0, 1],
            values: [2, 3]
        )
        let b = Tensor([[1, 0], [0, 1]])
        let result = CSRMatrix.matmul(a, b)
        #expect(result.shape == [3, 2])
        #expect(Array(result) == [2, 0, 0, 0, 0, 3])
    }

    @Test func matchesDense() {
        let a = CSRMatrix(
            rows: 2, columns: 3,
            rowPointers: [0, 2, 4],
            columnIndices: [0, 2, 1, 2],
            values: [1, 2, 3, 4]
        )
        let b = Tensor([[1, 2], [3, 4], [5, 6]])
        let sparse = CSRMatrix.matmul(a, b)
        let dense = Tensor.matmul(a.toTensor(), b)
        #expect(Array(sparse) == Array(dense))
    }

    @Test func nonSquare() {
        // A: 2x3, B: 3x4 -> 2x4
        let a = CSRMatrix(
            rows: 2, columns: 3,
            rowPointers: [0, 1, 2],
            columnIndices: [1, 0],
            values: [5, 3]
        )
        let b = Tensor(shape: [3, 4], elements: Array(1...12))
        let sparse = CSRMatrix.matmul(a, b)
        let dense = Tensor.matmul(a.toTensor(), b)
        #expect(sparse.shape == [2, 4])
        #expect(Array(sparse) == Array(dense))
    }
}

struct CSRMatrixSparseMatmulTests {
    @Test func basic() {
        // A: [[1, 0], [0, 2]]  B: [[3, 0], [0, 4]]
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1, 2]
        )
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [3, 4]
        )
        // row 0: 1*[3,0] = [3, 0]
        // row 1: 2*[0,4] = [0, 8]
        let result = CSRMatrix.matmul(a, b)
        #expect(result.rows == 2)
        #expect(result.columns == 2)
        #expect(result.rowPointers == [0, 1, 2])
        #expect(result.columnIndices == [0, 1])
        #expect(result.values == [3, 8])
    }

    @Test func identityMatrix() {
        let eye = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1, 1]
        )
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 2, 2],
            columnIndices: [0, 1],
            values: [10, 20]
        )
        let result = CSRMatrix.matmul(eye, b)
        #expect(result == b)
    }

    @Test func emptyOperand() {
        let a = CSRMatrix<Int>(rows: 2, columns: 3)
        let b = CSRMatrix(
            rows: 3, columns: 2,
            rowPointers: [0, 1, 2, 3],
            columnIndices: [0, 1, 0],
            values: [1, 2, 3]
        )
        let result = CSRMatrix.matmul(a, b)
        #expect(result.rows == 2)
        #expect(result.columns == 2)
        #expect(result.nnz == 0)
    }

    @Test func disjointStructure() {
        // A has columns 0; B has nonzeros in row 1 only
        // No overlap: A's column indices don't match B's nonzero rows
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 1],
            columnIndices: [0],
            values: [5]
        )
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 0, 1],
            columnIndices: [1],
            values: [3]
        )
        let result = CSRMatrix.matmul(a, b)
        #expect(result.nnz == 0)
    }

    @Test func matchesDense() {
        let a = CSRMatrix(
            rows: 2, columns: 3,
            rowPointers: [0, 2, 3],
            columnIndices: [0, 2, 1],
            values: [1, 2, 3]
        )
        let b = CSRMatrix(
            rows: 3, columns: 2,
            rowPointers: [0, 1, 2, 3],
            columnIndices: [0, 1, 0],
            values: [4, 5, 6]
        )
        let sparse = CSRMatrix.matmul(a, b)
        let dense = Tensor.matmul(a.toTensor(), b.toTensor())
        #expect(Array(sparse.toTensor()) == Array(dense))
    }

    @Test func cancellation() {
        // Products cancel out to zero for some entries
        // A: [[1, -1]], B: [[3], [3]]  ->  1*3 + (-1)*3 = 0
        let a = CSRMatrix(
            rows: 1, columns: 2,
            rowPointers: [0, 2],
            columnIndices: [0, 1],
            values: [1, -1]
        )
        let b = CSRMatrix(
            rows: 2, columns: 1,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 0],
            values: [3, 3]
        )
        let result = CSRMatrix.matmul(a, b)
        // Explicit zeros are kept (consistent with sparse arithmetic)
        #expect(result.toTensor()[0, 0] == 0)
    }

    @Test func nonSquare() {
        // A: 2x3, B: 3x4
        let a = CSRMatrix(
            rows: 2, columns: 3,
            rowPointers: [0, 1, 2],
            columnIndices: [1, 0],
            values: [5, 3]
        )
        let b = CSRMatrix(
            rows: 3, columns: 4,
            rowPointers: [0, 2, 3, 4],
            columnIndices: [0, 3, 1, 2],
            values: [1, 2, 3, 4]
        )
        let sparse = CSRMatrix.matmul(a, b)
        let dense = Tensor.matmul(a.toTensor(), b.toTensor())
        #expect(sparse.rows == 2)
        #expect(sparse.columns == 4)
        #expect(Array(sparse.toTensor()) == Array(dense))
    }
}
