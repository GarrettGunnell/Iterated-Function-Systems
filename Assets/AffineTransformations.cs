using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AffineTransformations : MonoBehaviour {

    [Serializable]
    public struct TransformInstructions {
        public Vector3 scale;
        public Vector3 shearX;
        public Vector3 shearY;
        public Vector3 shearZ;
        public Vector3 rotate;
        public Vector3 translate;
    }

    public List<TransformInstructions> transforms = new List<TransformInstructions>();
    
    private List<Matrix4x4> affineTransforms = new List<Matrix4x4>();

    public int GetTransformCount() {
        return affineTransforms.Count;
    }

    public Matrix4x4[] GetTransformData() {
        return affineTransforms.ToArray();
    }

    Matrix4x4 Scale(Vector3 s) {
        Matrix4x4 scaleMatrix = Matrix4x4.identity;

        scaleMatrix.SetRow(0, new Vector4(s.x, 0, 0, 0));
        scaleMatrix.SetRow(1, new Vector4(0, s.y, 0, 0));
        scaleMatrix.SetRow(2, new Vector4(0, 0, s.z, 0));

        return scaleMatrix;
    }

    Matrix4x4 ShearX(Vector3 s) {
        Matrix4x4 shearMatrix = Matrix4x4.identity;

        shearMatrix.SetRow(0, new Vector4(1, s.y, s.z, 0));
        shearMatrix.SetRow(1, new Vector4(0, 1, 0, 0));
        shearMatrix.SetRow(2, new Vector4(0, 0, 1, 0));

        return shearMatrix;
    }

    Matrix4x4 ShearY(Vector3 s) {
        Matrix4x4 shearMatrix = Matrix4x4.identity;

        shearMatrix.SetRow(0, new Vector4(1, 0, 0, 0));
        shearMatrix.SetRow(1, new Vector4(s.x, 1, s.z, 0));
        shearMatrix.SetRow(2, new Vector4(0, 0, 1, 0));

        return shearMatrix;
    }

    Matrix4x4 ShearZ(Vector3 s) {
        Matrix4x4 shearMatrix = Matrix4x4.identity;

        shearMatrix.SetRow(0, new Vector4(1, 0, 0, 0));
        shearMatrix.SetRow(1, new Vector4(0, 1, 0, 0));
        shearMatrix.SetRow(2, new Vector4(s.x, s.y, 1, 0));

        return shearMatrix;
    }

    Matrix4x4 Translate(Vector3 t) {
        Matrix4x4 transformMatrix = Matrix4x4.identity;

        transformMatrix.SetRow(0, new Vector4(1, 0, 0, t.x));
        transformMatrix.SetRow(1, new Vector4(0, 1, 0, t.y));
        transformMatrix.SetRow(2, new Vector4(0, 0, 1, t.z));

        return transformMatrix;
    }

    void AffineFromInstructions(TransformInstructions instructions) {
        Matrix4x4 affine = Matrix4x4.identity;

        Matrix4x4 scale = Scale(instructions.scale);
        Matrix4x4 shearX = ShearX(instructions.shearX);
        Matrix4x4 shearY = ShearY(instructions.shearY);
        Matrix4x4 shearZ = ShearZ(instructions.shearZ);
        Matrix4x4 shear = shearZ * shearY * shearX;
        Matrix4x4 translate = Translate(instructions.translate);

        affine = shear * scale * translate;

        affineTransforms.Add(affine);
    }

    void PopulateAffineBuffer() {
        affineTransforms.Clear();

        for (int i = 0; i < transforms.Count; ++i) {
            AffineFromInstructions(transforms[i]);
        }
    }

    void OnEnable() {
        PopulateAffineBuffer();
    }

    void Update() {
        PopulateAffineBuffer();
    }
}
