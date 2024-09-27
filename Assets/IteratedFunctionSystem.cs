using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections.LowLevel.Unsafe;

public class IteratedFunctionSystem : MonoBehaviour {
    public AffineTransformations affineTransformations;

    public Shader pointShader;

    public ComputeShader particleUpdater;

    public uint particlesPerBatch = 200000;
    public uint batchCount = 1;

    public bool uncapped = false;

    public Mesh[] pointCloudMeshes;
    private Material pointMaterial;

    private RenderParams pointRenderParams;

    [NonSerialized]
    public int threadsPerGroup = 64;

    void InitializeRenderParams() {
        pointRenderParams = new RenderParams(pointMaterial);
        pointRenderParams.worldBounds = new Bounds(Vector3.zero, 10000 * Vector3.one);
        // pointRenderParams.matProps = new MaterialPropertyBlock();
    }

    void InitializeMeshes() {
        pointCloudMeshes = new Mesh[batchCount];

        for (int i = 0; i < batchCount; ++i) {
            // Create point cloud mesh
            pointCloudMeshes[i] = new Mesh();

            pointCloudMeshes[i].vertexBufferTarget |= GraphicsBuffer.Target.Raw;
            pointCloudMeshes[i].indexBufferTarget |= GraphicsBuffer.Target.Raw;

            var vp = new VertexAttributeDescriptor(UnityEngine.Rendering.VertexAttribute.Position, VertexAttributeFormat.Float32, 3);

            pointCloudMeshes[i].SetVertexBufferParams((int)particlesPerBatch, vp);
            pointCloudMeshes[i].SetIndexBufferParams((int)particlesPerBatch, IndexFormat.UInt32);

            pointCloudMeshes[i].SetSubMesh(0, new SubMeshDescriptor(0, (int)particlesPerBatch, MeshTopology.Points), MeshUpdateFlags.DontRecalculateBounds);

            // Initialize point cloud vertices
            int cubeRootParticleCount = Mathf.CeilToInt(Mathf.Pow(particlesPerBatch, 1.0f / 3.0f));
            particleUpdater.SetInt("_CubeResolution", cubeRootParticleCount);
            particleUpdater.SetFloat("_CubeSize", 1.0f / cubeRootParticleCount);
            particleUpdater.SetInt("_ParticleCount", (int)particlesPerBatch);

            particleUpdater.SetBuffer(0, "_VertexBuffer", pointCloudMeshes[i].GetVertexBuffer(0));
            particleUpdater.SetBuffer(0, "_IndexBuffer", pointCloudMeshes[i].GetIndexBuffer());
            particleUpdater.Dispatch(0, Mathf.CeilToInt(particlesPerBatch / threadsPerGroup), 1, 1);
        }
    }


    void OnEnable() {
        // Debug.Log(SystemInfo.graphicsDeviceName);

        // UnsafeUtility.SetLeakDetectionMode(Unity.Collections.NativeLeakDetectionMode.Enabled);
        pointMaterial = new Material(pointShader);

        InitializeMeshes();
        InitializeRenderParams();
    }
    
    public virtual void IterateSystem() {
        for (int i = 0; i < batchCount; ++i) {
            particleUpdater.SetInt("_TransformationCount", affineTransformations.GetTransformCount());
            particleUpdater.SetInt("_Seed", Mathf.CeilToInt(UnityEngine.Random.Range(1, 1000000)));
            particleUpdater.SetBuffer(2, "_VertexBuffer", pointCloudMeshes[i].GetVertexBuffer(0));
            particleUpdater.SetBuffer(2, "_Transformations", affineTransformations.GetAffineBuffer());
            particleUpdater.Dispatch(2, Mathf.CeilToInt(particlesPerBatch / threadsPerGroup), 1, 1);
        }
    }

    void DrawParticles() {
        for (int i = 0; i < batchCount; ++i) {
            Graphics.RenderMesh(pointRenderParams, pointCloudMeshes[i], 0, Matrix4x4.Scale(new Vector3(2.0f, 2.0f, 2.0f)));
        }
    }

    void Update() {
        if (uncapped) {
            IterateSystem();
        } else {
            if (Input.GetKeyDown("space")) {
                IterateSystem();
            }
        }

        DrawParticles();
    }

    void OnDisable() {
        for (int i = 0; i < batchCount; ++i) {
            var vertBuffer = pointCloudMeshes[i].GetVertexBuffer(0);
            var indexBuffer = pointCloudMeshes[i].GetIndexBuffer();

            vertBuffer.Release();
            indexBuffer.Release();
            UnityEngine.Object.Destroy(pointCloudMeshes[i]);
        }
    }
}
