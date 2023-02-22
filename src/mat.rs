use core::ops::Add;
use std::fmt::Debug;
use std::ops::AddAssign;
use nalgebra::DMatrix;

/// matrix add with broadcasting
/// like the numpy add operation
/// not sure I even need it anymore, after fixing inconsistencies with the matrix shapes
/// TODO see if it's still needed, or that standard matrix addition suffices
pub fn add<T>(v1: DMatrix<T>, v2: DMatrix<T>) -> Result<DMatrix<T>, String>
    where T: PartialEq + Copy + Clone + Debug + Add + Add<Output=T> + AddAssign + 'static
{
    let (r1, c1) = v1.shape();
    let (r2, c2) = v2.shape();

    if r1 == r2 && c1 == c2 {
        // same size, no broadcasting needed
        Ok(v1 + v2)
    } else if r1 == 1 && c2 == 1 {
        Ok(DMatrix::from_fn(r2, c1, |r, c| *v1.get(c).unwrap() + *v2.get(r).unwrap()))
    } else if c1 == 1 && r2 == 1 {
        Ok(DMatrix::from_fn(r1, c2, |r, c| *v1.get(r).unwrap() + *v2.get(c).unwrap()))
    } else if r1 == 1 && c1 == c2 {
        Ok(DMatrix::from_fn(r2, c1, |r, c| *v1.get(c).unwrap() + *v2.get(c * r2 + r).unwrap()))
    } else if r2 == 1 && c1 == c2 {
        Ok(DMatrix::from_fn(r1, c2, |r, c| *v2.get(c).unwrap() + *v1.get(c * r1 + r).unwrap()))
    } else if c1 == 1 && r1 == r2 {
        Ok(DMatrix::from_fn(r1, c2, |r, c| *v1.get(r).unwrap() + *v2.get(c * r2 + r).unwrap()))
    } else if c2 == 1 && r1 == r2 {
        Ok(DMatrix::from_fn(r2, c1, |r, c| *v2.get(r).unwrap() + *v1.get(c * r2 + r).unwrap()))
    } else {
        Err(format!("ValueError: operands could not be broadcast together ({},{}), ({},{})", r1,c1, r2, c2))
    }
}

#[cfg(test)]
mod test {
    use nalgebra::dmatrix;
    use super::*;

    #[test]
    fn stretch_row_column_to_square() {
        let v1: DMatrix<u32> = dmatrix![1,2,3];
        let v2: DMatrix<u32> = dmatrix![1;2;3];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum.shape(), (3, 3));
        assert_eq!(sum, dmatrix![2,3,4;3,4,5;4,5,6]);
    }

    #[test]
    fn stretch_row_column_to_rect() {
        let v1: DMatrix<u32> = dmatrix![1,2,3];
        let v2: DMatrix<u32> = dmatrix![1;2];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum.shape(), (2, 3));
        assert_eq!(sum, dmatrix![2,3,4;3,4,5]);
    }

    #[test]
    fn stretch_column_row_to_square() {
        let v1: DMatrix<u32> = dmatrix![1;2;3];
        let v2: DMatrix<u32> = dmatrix![1,2,3];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum.shape(), (3, 3));
        assert_eq!(sum, dmatrix![2,3,4;3,4,5;4,5,6]);
    }

    #[test]
    fn stretch_column_row_to_rect() {
        let v1: DMatrix<u32> = dmatrix![1;2;3];
        let v2: DMatrix<u32> = dmatrix![1,2];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum.shape(), (3, 2));
        assert_eq!(sum, dmatrix![2,3;3,4;4,5]);
    }

    #[test]
    fn stretch_row() {
        let v1: DMatrix<u32> = dmatrix![1,2,3];
        let v2: DMatrix<u32> = dmatrix![1,2,3;4,5,6];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum.shape(), (2, 3));
        assert_eq!(sum, dmatrix![2,4,6;5,7,9]);
    }

    #[test]
    fn stretch_row_commute() {
        let v1: DMatrix<u32> = dmatrix![1,2,3;4,5,6];
        let v2: DMatrix<u32> = dmatrix![1,2,3];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum.shape(), (2, 3));
        assert_eq!(sum, dmatrix![2,4,6;5,7,9]);
    }

    #[test]
    fn stretch_column() {
        let v1: DMatrix<u32> = dmatrix![1;2];
        let v2: DMatrix<u32> = dmatrix![1,2,3;4,5,6];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum.shape(), (2, 3));
        assert_eq!(sum, dmatrix![2,3,4;6,7,8]);
    }

    #[test]
    fn stretch_column_commute() {
        let v1: DMatrix<u32> = dmatrix![1,2,3;4,5,6];
        let v2: DMatrix<u32> = dmatrix![1;2];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum.shape(), (2, 3));
        assert_eq!(sum, dmatrix![2,3,4;6,7,8]);
    }

    #[test]
    fn test_broadcast_2dims() {
        let v1: DMatrix<u32> = dmatrix![1,2,3];
        let v2: DMatrix<u32> = dmatrix![1;2;3];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum.shape(), (3, 3));
        assert_eq!(sum, dmatrix![2,3,4;3,4,5;4,5,6]);
    }

    #[test]
    fn test_add_commutative() {
        let v1 = dmatrix![1,2,3];
        let v2 = dmatrix![1;2;3];

        let sum = add(v2, v1).unwrap();
        assert_eq!(sum.shape(), (3, 3));
        assert_eq!(sum, dmatrix![2,3,4;3,4,5;4,5,6]);
    }

    #[test]
    fn test_add_same_size() {
        let v1 = dmatrix![1,2;3,4];
        let v2 = dmatrix![3,4;5,6];

        let sum = add(v2, v1).unwrap();
        assert_eq!(sum.shape(), (2, 2));
        assert_eq!(sum, dmatrix![4,6;8,10]);
    }

    #[test]
    fn test_add_row_broadcast() {//
        let v1 = dmatrix![1,2;3,4];
        let v2 = dmatrix![3,4];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum, dmatrix![4,6;6,8]);
    }

    #[test]
    fn test_add_row_broadcast2() {
        let v1 = dmatrix![1,1];
        let v2 = dmatrix![1,2;3,4];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum.shape(), (2, 2));
        assert_eq!(sum, dmatrix![2,3;4,5]);
    }

    #[test]
    fn test_column_broadcast() {
        let v1 = dmatrix![1;1];
        let v2 = dmatrix![1,2;3,4];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum.shape(), (2, 2));
        assert_eq!(sum, dmatrix![2,3;4,5]);
    }

    #[test]
    fn test_column_broadcast2() {
        let v1 = dmatrix![1,2;3,4];
        let v2 = dmatrix![1;1];

        let sum = add(v1, v2).unwrap();
        assert_eq!(sum.shape(), (2, 2));
        assert_eq!(sum, dmatrix![2,3;4,5]);
    }

    #[test]
    fn column_too_long() {
        let v1 = dmatrix![1;1;1];
        let v2 = dmatrix![1,2;3,4];

        let result = add(v1, v2);
        assert_eq!(result, Err("ValueError: operands could not be broadcast together".to_owned()));
    }

    #[test]
    fn row_too_long() {
        let v1 = dmatrix![1,1,1];
        let v2 = dmatrix![1,2;3,4];

        let result = add(v1, v2);
        assert_eq!(result, Err("ValueError: operands could not be broadcast together".to_owned()));
    }


}