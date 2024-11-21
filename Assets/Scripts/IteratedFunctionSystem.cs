using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections.LowLevel.Unsafe;

public class IteratedFunctionSystem : MonoBehaviour {
    public AffineTransformations affineTransformations;

    public Shader instancedPointShader;

    public ComputeShader particleUpdater, parallelReducer;

    public uint particlesPerBatch = 200000;
    public uint batchCount = 1;
    public bool updateInstanceCount = true;

    public uint lowDetailGenerations = 8;
    private uint lowDetailParticleCount = 0;

    public bool viewLowDetail = false;

    public bool uncapped = false;

    public Mesh[] pointCloudMeshes;
    private Mesh lowDetailMesh;

    private Material instancedPointMaterial;

    private RenderParams instancedRenderParams;

    [NonSerialized]
    public int threadsPerGroup = 64;

    public bool predictOrigin = false;

    public enum BoundsCalculationMode {
        SingleThreadedScan = 0,
        DivergentBranching,
        BankConflict,
        SequentialAddressing,
        UnrollLastWarp
    }; public BoundsCalculationMode boundsCalculationMode;
    private BoundsCalculationMode cachedCalculationMode;

    public bool toggleDoubleLoad = true;

    [Range(0.0f, 5.0f)]
    public float scalePadding = 1.0f;

    GraphicsBuffer instancedCommandBuffer, lowDetailCommandBuffer;
    GraphicsBuffer.IndirectDrawIndexedArgs[] instancedCommandIndexedData;

    private ComputeBuffer[] lowDetailIterationBuffers;
    private ComputeBuffer reductionDataBuffer, finalTransformBuffer;
    private ComputeBuffer reductionBuffer;

    private ComputeShader minReducer, maxReducer, addReducer;

    private int reductionGroupSize = 128;

    private float newScale;
    private Vector3 newOrigin;

    private bool buffersInitialized = false;

    void InitializeRenderParams() {
        instancedRenderParams = new RenderParams(instancedPointMaterial);
        instancedRenderParams.worldBounds = new Bounds(Vector3.zero, 10000 * Vector3.one);
        instancedRenderParams.matProps = new MaterialPropertyBlock();

        instancedCommandBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, GraphicsBuffer.IndirectDrawIndexedArgs.size);
        instancedCommandIndexedData = new GraphicsBuffer.IndirectDrawIndexedArgs[1];
        instancedCommandIndexedData[0].instanceCount = System.Convert.ToUInt32(affineTransformations.GetTransformCount());
        instancedCommandIndexedData[0].indexCountPerInstance = particlesPerBatch;

        instancedCommandBuffer.SetData(instancedCommandIndexedData);

        lowDetailCommandBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, GraphicsBuffer.IndirectDrawIndexedArgs.size);
        instancedCommandIndexedData[0].indexCountPerInstance = lowDetailParticleCount;

        lowDetailCommandBuffer.SetData(instancedCommandIndexedData);
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

    public Shader voxelShader;
    public Mesh voxelMesh;
    GraphicsBuffer voxelGrid, occlusionGrid, commandBuffer;
    GraphicsBuffer.IndirectDrawIndexedArgs[] commandIndexedData;
    public int voxelBounds;
    public float voxelSize;
    
    [Range(1, 32)]
    public int meshesToVoxelize = 1;

    public bool useLowDetailForVoxels = false;

    public bool renderVoxels = false;
    private int voxelDimension, voxelCount;
    private Material voxelMaterial;
    private RenderParams renderParams;


    void InitializeVoxelGrid() {
        voxelMaterial = new Material(voxelShader);

        voxelDimension = Mathf.FloorToInt(voxelBounds / voxelSize);
        voxelCount = voxelDimension * voxelDimension * voxelDimension;

        Debug.Log("Grid Dimension: " + voxelDimension.ToString());
        Debug.Log("Voxel Count: " + voxelCount.ToString());

        voxelGrid = new GraphicsBuffer(GraphicsBuffer.Target.Structured, voxelCount, System.Runtime.InteropServices.Marshal.SizeOf(typeof(int)));
        occlusionGrid = new GraphicsBuffer(GraphicsBuffer.Target.Structured, voxelCount, System.Runtime.InteropServices.Marshal.SizeOf(typeof(float)));

        renderParams = new RenderParams(voxelMaterial);
        renderParams.worldBounds = new Bounds(Vector3.zero, 10000 * Vector3.one);
        renderParams.matProps = new MaterialPropertyBlock();

        renderParams.matProps.SetBuffer("_VoxelGrid", voxelGrid);
        renderParams.matProps.SetBuffer("_OcclusionGrid", occlusionGrid);
        renderParams.matProps.SetFloat("_VoxelSize", voxelSize);
        renderParams.matProps.SetInt("_GridSize", voxelDimension);
        renderParams.matProps.SetInt("_GridBounds", voxelBounds);

        instancedRenderParams.matProps.SetBuffer("_OcclusionGrid", occlusionGrid);
        instancedRenderParams.matProps.SetInt("_GridSize", voxelDimension);
        instancedRenderParams.matProps.SetInt("_GridBounds", voxelBounds);

        commandBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, GraphicsBuffer.IndirectDrawIndexedArgs.size);
        commandIndexedData = new GraphicsBuffer.IndirectDrawIndexedArgs[1];
        commandIndexedData[0].instanceCount = System.Convert.ToUInt32(voxelCount);
        commandIndexedData[0].indexCountPerInstance = voxelMesh.GetIndexCount(0);

        commandBuffer.SetData(commandIndexedData);
    }

    void InitializePredictedTransform() {
        reductionDataBuffer = new ComputeBuffer(3, System.Runtime.InteropServices.Marshal.SizeOf(typeof(Vector3)));
        finalTransformBuffer = new ComputeBuffer(1, System.Runtime.InteropServices.Marshal.SizeOf(typeof(Matrix4x4)));

        lowDetailParticleCount = (uint)Mathf.CeilToInt(Mathf.Pow(affineTransformations.GetTransformCount(), lowDetailGenerations));
        Debug.Log("Transform Count: " + affineTransformations.GetTransformCount().ToString());
        Debug.Log("Particle Count: " + lowDetailParticleCount.ToString());
        lowDetailIterationBuffers = new ComputeBuffer[lowDetailGenerations - 1];

        for (int i = 0; i < lowDetailGenerations - 1; ++i) {
            int bufferSize = Mathf.CeilToInt(Mathf.Pow(affineTransformations.GetTransformCount(), i + 1));

            Debug.Log("Buffer " + i.ToString() + ": " + bufferSize.ToString());

            lowDetailIterationBuffers[i] = new ComputeBuffer(bufferSize, System.Runtime.InteropServices.Marshal.SizeOf(typeof(Vector3)));
        }
        
        Debug.Log("Low Detail Vertices: " + lowDetailParticleCount.ToString());

        lowDetailMesh = new Mesh();

        lowDetailMesh.vertexBufferTarget |= GraphicsBuffer.Target.Raw;
        lowDetailMesh.indexBufferTarget |= GraphicsBuffer.Target.Raw;

        var vp = new VertexAttributeDescriptor(UnityEngine.Rendering.VertexAttribute.Position, VertexAttributeFormat.Float32, 3);

        lowDetailMesh.SetVertexBufferParams((int)lowDetailParticleCount, vp);
        lowDetailMesh.SetIndexBufferParams((int)lowDetailParticleCount, IndexFormat.UInt32);

        lowDetailMesh.SetSubMesh(0, new SubMeshDescriptor(0, (int)lowDetailParticleCount, MeshTopology.Points), MeshUpdateFlags.DontRecalculateBounds);


        int totalReductionGroups = Mathf.CeilToInt(lowDetailParticleCount / 128);
        reductionBuffer = new ComputeBuffer(totalReductionGroups, System.Runtime.InteropServices.Marshal.SizeOf(typeof(Vector3)));

        minReducer = Instantiate(parallelReducer);
        minReducer.EnableKeyword("MIN_REDUCTION");
        minReducer.DisableKeyword("MAX_REDUCTION");
        minReducer.DisableKeyword("ADD_REDUCTION");

        maxReducer = Instantiate(parallelReducer);
        maxReducer.EnableKeyword("MAX_REDUCTION");
        maxReducer.DisableKeyword("MIN_REDUCTION");
        maxReducer.DisableKeyword("ADD_REDUCTION");

        addReducer = Instantiate(parallelReducer);
        addReducer.DisableKeyword("MAX_REDUCTION");
        addReducer.DisableKeyword("MIN_REDUCTION");
        addReducer.EnableKeyword("ADD_REDUCTION");

    }

    void EnableReductionKeyword(ComputeShader reducer, BoundsCalculationMode boundsMode) {
        if (boundsMode == BoundsCalculationMode.DivergentBranching)
            reducer.EnableKeyword("INTERLEAVED_ADDRESSING_DIVERGENT");
        if (boundsMode == BoundsCalculationMode.BankConflict)
            reducer.EnableKeyword("INTERLEAVED_ADDRESSING_BANK_CONFLICT");
        if (boundsMode == BoundsCalculationMode.SequentialAddressing)
            reducer.EnableKeyword("SEQUENTIAL_ADDRESSING");
        if (boundsMode == BoundsCalculationMode.UnrollLastWarp)
            reducer.EnableKeyword("UNROLL_LAST_WARP");
    }

    void DisableReductionKeywords(ComputeShader reducer) {
        reducer.DisableKeyword("INTERLEAVED_ADDRESSING_DIVERGENT");
        reducer.DisableKeyword("INTERLEAVED_ADDRESSING_BANK_CONFLICT");
        reducer.DisableKeyword("SEQUENTIAL_ADDRESSING");
        reducer.DisableKeyword("UNROLL_LAST_WARP");
    }

    void UpdateReductionKeywords(ComputeShader reducer) {
        DisableReductionKeywords(reducer);
        EnableReductionKeyword(reducer, boundsCalculationMode);
        cachedCalculationMode = boundsCalculationMode;
    }

    void ToggleDoubleLoad() {
        if (toggleDoubleLoad) {
            if (minReducer.IsKeywordEnabled("DOUBLE_LOAD")) {
                minReducer.DisableKeyword("DOUBLE_LOAD");
                maxReducer.DisableKeyword("DOUBLE_LOAD");
                addReducer.DisableKeyword("DOUBLE_LOAD");
                reductionGroupSize = 128;
                Debug.Log("Disabled double load");
            } else {
                minReducer.EnableKeyword("DOUBLE_LOAD");
                maxReducer.EnableKeyword("DOUBLE_LOAD");
                addReducer.EnableKeyword("DOUBLE_LOAD");
                reductionGroupSize = 256;
                Debug.Log("Enabled double load");
            }
            toggleDoubleLoad = false;
        }
    }


    void OnEnable() {
        // Debug.Log(SystemInfo.graphicsDeviceName);

        buffersInitialized = false;

        Application.targetFrameRate = 120;

        UnsafeUtility.SetLeakDetectionMode(Unity.Collections.NativeLeakDetectionMode.Enabled);
        instancedPointMaterial = new Material(instancedPointShader);

        if (affineTransformations.GetTransformCount() != 0) {

            Debug.Log("Enabling");
            InitializeMeshes();
            InitializePredictedTransform();
            InitializeRenderParams();
            InitializeVoxelGrid();

            cachedCalculationMode = boundsCalculationMode;
            UpdateReductionKeywords(minReducer);
            UpdateReductionKeywords(maxReducer);
            UpdateReductionKeywords(addReducer);
            ToggleDoubleLoad();

            buffersInitialized = true;
        }
    }
    
    public virtual void IterateSystem() {
        // Reset System
        int cubeRootParticleCount = Mathf.CeilToInt(Mathf.Pow(particlesPerBatch, 1.0f / 3.0f));
        particleUpdater.SetInt("_CubeResolution", cubeRootParticleCount);
        particleUpdater.SetFloat("_CubeSize", 0);

        particleUpdater.SetBuffer(0, "_VertexBuffer", pointCloudMeshes[0].GetVertexBuffer(0));
        particleUpdater.SetBuffer(0, "_IndexBuffer", pointCloudMeshes[0].GetIndexBuffer());
        particleUpdater.Dispatch(0, Mathf.CeilToInt(particlesPerBatch / threadsPerGroup), 1, 1);

        // Seed First Iteration
        int transformCount = affineTransformations.GetTransformCount();

        particleUpdater.SetInt("_TransformationCount", transformCount);
        particleUpdater.SetInt("_GenerationOffset", 0);
        particleUpdater.SetInt("_GenerationLimit", transformCount);
        particleUpdater.SetBuffer(2, "_VertexBuffer", pointCloudMeshes[0].GetVertexBuffer(0));
        particleUpdater.SetBuffer(2, "_Transformations", affineTransformations.GetAffineBuffer());
        particleUpdater.Dispatch(2, Mathf.CeilToInt(particlesPerBatch / threadsPerGroup), 1, 1);

        int iteratedParticles = transformCount;
        int previousGenerationSize = transformCount;
        while (iteratedParticles < particlesPerBatch) {
            int generationSize = previousGenerationSize * transformCount;

            particleUpdater.SetInt("_GenerationOffset", iteratedParticles);
            particleUpdater.SetInt("_GenerationLimit", (int)Mathf.Clamp(iteratedParticles + generationSize, 0, particlesPerBatch));

            particleUpdater.SetBuffer(2, "_VertexBuffer", pointCloudMeshes[0].GetVertexBuffer(0));
            particleUpdater.SetBuffer(2, "_Transformations", affineTransformations.GetAffineBuffer());

            
            particleUpdater.Dispatch(2, Mathf.CeilToInt(particlesPerBatch / threadsPerGroup), 1, 1);
            

            iteratedParticles += generationSize;
            previousGenerationSize = generationSize;
        }
    }

    Matrix4x4 GetFinalFinalTransform() {
        if (predictOrigin) {
            return  Matrix4x4.Scale((Vector3.one * voxelBounds) / newScale) * Matrix4x4.Translate(-newOrigin);
        }

        return affineTransformations.GetFinalTransform();
    }

    void Reduce(ComputeShader reducer) {
        int reductionGroupCount = Mathf.CeilToInt(lowDetailParticleCount / reductionGroupSize);

        // Initial Reduction (Mesh -> Reduction Buffer)
        reducer.SetBuffer(1, "_InputBuffer", lowDetailMesh.GetVertexBuffer(0));
        reducer.SetBuffer(1, "_OutputBuffer", reductionBuffer);
        reducer.SetInt("_ReductionBufferSize", (int)lowDetailParticleCount);
        reducer.Dispatch(1, reductionGroupCount, 1, 1);
        
        // Cross-kernel Reduction (Reduction Buffer -> Reduction Buffer)
        while (reductionGroupCount > reductionGroupSize) {
            reductionGroupCount = Mathf.CeilToInt(reductionGroupCount / reductionGroupSize);

            // Global Min Reduce
            reducer.SetBuffer(1, "_InputBuffer", reductionBuffer);
            reducer.SetBuffer(1, "_OutputBuffer", reductionBuffer);
            reducer.SetInt("_ReductionBufferSize", reductionGroupCount);
            reducer.Dispatch(1, reductionGroupCount, 1, 1);
        }

        // Final Reduction (Reduction Buffer -> Bounding Box Buffer)
        reducer.SetInt("_ReductionBufferSize", Math.Min(128, reductionGroupCount));
        reducer.SetBuffer(2, "_InputBuffer", reductionBuffer);
        reducer.SetBuffer(2, "_OutputBuffer", reductionDataBuffer);
        reducer.Dispatch(2, 1, 1, 1);
    }

    List<Vector3> gizmoPoints = new();

    public bool dumpData = false;
    public bool updateGizmoPoints = false;
    void PredictFinalTransform() {
        // Seed First Iteration
        int transformCount = affineTransformations.GetTransformCount();

        parallelReducer.SetInt("_TransformationCount", transformCount);
        parallelReducer.SetBuffer(4, "_OutputBuffer", lowDetailIterationBuffers[0]);
        parallelReducer.SetBuffer(4, "_Transformations", affineTransformations.GetAffineBuffer());

        parallelReducer.Dispatch(4, 1, 1, 1); // Assumes transform count <64

        for (int i = 1; i < lowDetailGenerations - 1; ++i) {
            parallelReducer.SetBuffer(5, "_InputBuffer", lowDetailIterationBuffers[i - 1]);
            parallelReducer.SetBuffer(5, "_OutputBuffer", lowDetailIterationBuffers[i]);
            parallelReducer.SetBuffer(5, "_Transformations", affineTransformations.GetAffineBuffer());

            int bufferSize = Mathf.CeilToInt(Mathf.Pow(affineTransformations.GetTransformCount(), i + 1));
            parallelReducer.Dispatch(5, (int)Mathf.Max(1, Mathf.CeilToInt(bufferSize / threadsPerGroup)), 1, 1);

        }
        
        parallelReducer.SetBuffer(5, "_InputBuffer", lowDetailIterationBuffers[lowDetailGenerations - 2]);
        parallelReducer.SetBuffer(5, "_OutputBuffer", lowDetailMesh.GetVertexBuffer(0));
        parallelReducer.SetBuffer(5, "_Transformations", affineTransformations.GetAffineBuffer());

        parallelReducer.Dispatch(5, Mathf.CeilToInt(lowDetailParticleCount / threadsPerGroup), 1, 1);


        if (boundsCalculationMode == BoundsCalculationMode.SingleThreadedScan) {
            parallelReducer.SetBuffer(0, "_InputBuffer", lowDetailMesh.GetVertexBuffer(0));
            parallelReducer.SetBuffer(0, "_OutputBuffer", reductionDataBuffer);
            parallelReducer.SetInt("_ReductionBufferSize", (int)lowDetailParticleCount);
            parallelReducer.Dispatch(0, 1, 1, 1);
        } else {
            Reduce(minReducer);
            Reduce(maxReducer);
            Reduce(addReducer);
        }

        parallelReducer.SetBuffer(3, "_InputBuffer", reductionDataBuffer);
        parallelReducer.SetBuffer(3, "_FinalTransformBuffer", finalTransformBuffer);
        parallelReducer.SetFloat("_TargetBoundsSize", voxelBounds);
        parallelReducer.SetFloat("_ScalePadding", scalePadding);
        parallelReducer.SetFloat("_ParticleCount", lowDetailParticleCount);
        parallelReducer.Dispatch(3, 1, 1, 1);

        if (updateGizmoPoints) {
            gizmoPoints.Clear();
            Vector3[] predictedPoints = new Vector3[3];

            reductionDataBuffer.GetData(predictedPoints);

            for (int i = 0; i < 3; ++i) {
                gizmoPoints.Add(new Vector3(predictedPoints[i].x, predictedPoints[i].y, predictedPoints[i].z));
            }

            Vector3 p1 = gizmoPoints[0];
            Vector3 p2 = gizmoPoints[1];
            Vector3 p3 = gizmoPoints[2];
        
            float x = Mathf.Abs(p1.x - p2.x);
            float y = Mathf.Abs(p1.y - p2.y);
            float z = Mathf.Abs(p1.z - p2.z);

            newOrigin = new Vector3(predictedPoints[2].x, predictedPoints[2].y, predictedPoints[2].z);
            newOrigin = Vector3.Lerp(p1, p2, 0.5f);
            newOrigin = p3 / lowDetailParticleCount;
            newScale = Mathf.Max(Vector3.Distance(p1, newOrigin), Vector3.Distance(p2, newOrigin));
            // newScale = Vector3.Distance(p1, p2);
            newScale *= scalePadding;
        }

        if (dumpData) {
            Vector3[] predictedPoints = new Vector3[3];

            reductionDataBuffer.GetData(predictedPoints);

            Vector3[] meshPoints = new Vector3[lowDetailParticleCount];
            lowDetailMesh.GetVertexBuffer(0).GetData(meshPoints);

            Vector3 cpuMin = Vector3.one * 1000000000;
            Vector3 cpuMax = Vector3.one * -1000000000;
            Vector3 cpuAdd = Vector3.zero;

            for (int i = 0; i < meshPoints.Length; ++i) {
                cpuMin = Vector3.Min(cpuMin, meshPoints[i]);
                cpuMax = Vector3.Max(cpuMax, meshPoints[i]);
                cpuAdd += meshPoints[i];
            }

            Debug.Log("Minimum Found By CPU: " + cpuMin.ToString());
            Debug.Log("Maximum Found By CPU: " + cpuMax.ToString());
            Debug.Log("Sum Found By CPU: " + cpuAdd.ToString());

            Debug.Log("Minimum Found By GPU: " + predictedPoints[0].ToString());
            Debug.Log("Maximum Found By GPU: " + predictedPoints[1].ToString());
            Debug.Log("Sum Found By GPU: " + predictedPoints[2].ToString());

            foreach (var localKeyword in minReducer.enabledKeywords) {
                Debug.Log("Local min shader keyword " + localKeyword.name + " is currently enabled");
            }
            foreach (var localKeyword in maxReducer.enabledKeywords) {
                Debug.Log("Local max shader keyword " + localKeyword.name + " is currently enabled");
            }
            foreach (var localKeyword in addReducer.enabledKeywords) {
                Debug.Log("Local add shader keyword " + localKeyword.name + " is currently enabled");
            }

            dumpData = false;
        }
    }

    void Voxelize() {
        int maxGroups = 65535;
        int maxThreadsPerGroup = threadsPerGroup * 65535;
        int groupCount = Mathf.CeilToInt(voxelCount / threadsPerGroup);

        int clearedMemoryOffset = 0;
        int clearedGroupCount = groupCount;
        while (clearedMemoryOffset < voxelCount) {
            particleUpdater.SetInt("_MemoryOffset", clearedMemoryOffset);

            // Clear Voxel Grid
            particleUpdater.SetBuffer(3, "_VoxelGrid", voxelGrid);
            particleUpdater.Dispatch(3, Mathf.Min(clearedGroupCount, maxGroups), 1, 1);

            // Clear Occlusion
            particleUpdater.SetBuffer(5, "_OcclusionGrid", occlusionGrid);
            particleUpdater.Dispatch(5, Mathf.Min(clearedGroupCount, maxGroups), 1, 1);

            clearedGroupCount -= maxGroups;
            clearedMemoryOffset += maxThreadsPerGroup;
        }

        // Particles To Voxel (Brute Force)
        particleUpdater.SetInt("_GridSize", Mathf.FloorToInt(voxelBounds / voxelSize));
        particleUpdater.SetInt("_GridBounds", voxelBounds);
        particleUpdater.SetBuffer(4, "_FinalTransformBuffer", finalTransformBuffer);
        particleUpdater.SetInt("_TransformationCount", affineTransformations.GetTransformCount());
        
        if (useLowDetailForVoxels) {
            particleUpdater.SetBuffer(4, "_VertexBuffer", lowDetailMesh.GetVertexBuffer(0));
            particleUpdater.SetBuffer(4, "_VoxelGrid", voxelGrid);
            particleUpdater.SetBuffer(4, "_Transformations", affineTransformations.GetAffineBuffer());
            particleUpdater.Dispatch(4, Mathf.CeilToInt(lowDetailParticleCount / threadsPerGroup), 1, 1);
        }
        else {
            for (int i = 0; i < Mathf.Min(meshesToVoxelize, batchCount); ++i) {
                particleUpdater.SetBuffer(4, "_VertexBuffer", pointCloudMeshes[i].GetVertexBuffer(0));
                particleUpdater.SetBuffer(4, "_VoxelGrid", voxelGrid);
                particleUpdater.SetBuffer(4, "_Transformations", affineTransformations.GetAffineBuffer());
                particleUpdater.Dispatch(4, Mathf.CeilToInt(particlesPerBatch / threadsPerGroup), 1, 1);
            }
        }

        int occlusionMemoryOffset = 0;
        int occlusionGroupCount = groupCount;
        particleUpdater.SetBuffer(6, "_VoxelGrid", voxelGrid);
        particleUpdater.SetBuffer(6, "_OcclusionGrid", occlusionGrid);
        while (occlusionMemoryOffset < voxelCount) {
            particleUpdater.SetInt("_MemoryOffset", occlusionMemoryOffset);
            particleUpdater.Dispatch(6, Mathf.Min(occlusionGroupCount, maxGroups), 1, 1);

            occlusionGroupCount -= maxGroups;
            occlusionMemoryOffset += maxThreadsPerGroup;
        }
    }

    [Range(0.0f, 3.0f)]
    public float occlusionMultiplier = 1.0f;

    [Range(0.0f, 5.0f)]
    public float occlusionAttenuation = 1.0f;

    public Color particleColor, occlusionColor;

    void DrawParticles() {
        instancedRenderParams.matProps.SetFloat("_OcclusionMultiplier", occlusionMultiplier);
        instancedRenderParams.matProps.SetFloat("_OcclusionAttenuation", occlusionAttenuation);
        instancedRenderParams.matProps.SetVector("_ParticleColor", particleColor);
        instancedRenderParams.matProps.SetVector("_OcclusionColor", occlusionColor);
        instancedRenderParams.matProps.SetBuffer("_FinalTransformBuffer", finalTransformBuffer);
        instancedRenderParams.matProps.SetBuffer("_Transformations", affineTransformations.GetAffineBuffer());

        if (viewLowDetail) {
            Graphics.RenderMeshIndirect(instancedRenderParams, lowDetailMesh, lowDetailCommandBuffer, 1);
        } else {
            for (int i = 0; i < batchCount; ++i) {
                Graphics.RenderMeshIndirect(instancedRenderParams, pointCloudMeshes[i], instancedCommandBuffer, 1);
            }
        }
    }

    void Update() {
        // Wait for affine transformations to create data buffers
        if (affineTransformations.GetTransformCount() == 0) return;

        if (!buffersInitialized) {
            InitializeMeshes();
            InitializePredictedTransform();
            InitializeRenderParams();
            InitializeVoxelGrid();

            cachedCalculationMode = boundsCalculationMode;
            UpdateReductionKeywords(minReducer);
            UpdateReductionKeywords(maxReducer);
            UpdateReductionKeywords(addReducer);
            ToggleDoubleLoad();

            buffersInitialized = true;
        }

        // Some weird race condition makes it so that the correct transformation count doesn't make it here in time so it breaks the instancing
        // As a hack, the first 1 second of runtime will repeatedly set this value in order to ensure proper functionality. In the industry, we call this a "loading screen"
        if (updateInstanceCount) {
            instancedCommandIndexedData[0].instanceCount = System.Convert.ToUInt32(affineTransformations.GetTransformCount());
            instancedCommandIndexedData[0].indexCountPerInstance = particlesPerBatch;

            instancedCommandBuffer.SetData(instancedCommandIndexedData);
            instancedCommandIndexedData[0].indexCountPerInstance = lowDetailParticleCount;

            lowDetailCommandBuffer.SetData(instancedCommandIndexedData);
            updateInstanceCount = false;

            Debug.Log("Particles in memory: " + (particlesPerBatch * batchCount).ToString());
            Debug.Log("Particles drawn with instancing: " + (particlesPerBatch * batchCount * instancedCommandIndexedData[0].instanceCount).ToString());
        }

        if (cachedCalculationMode != boundsCalculationMode) {
            UpdateReductionKeywords(minReducer);
            UpdateReductionKeywords(maxReducer);
            UpdateReductionKeywords(addReducer);
        }

        ToggleDoubleLoad();

        // Enable/Disable system iteration
        if (Input.GetKeyDown("r")) uncapped = !uncapped;

        if (uncapped && affineTransformations.GetTransformCount() != 0) {
            PredictFinalTransform();
            IterateSystem();
        }

        Voxelize();

        if (renderVoxels) {
            Graphics.RenderMeshIndirect(renderParams, voxelMesh, commandBuffer, 1);
        } else {
            DrawParticles();
        }
    }

    void OnDisable() {
        for (int i = 0; i < batchCount; ++i) {
            pointCloudMeshes[i].GetVertexBuffer(0).Release();
            pointCloudMeshes[i].GetIndexBuffer().Release();

            UnityEngine.Object.Destroy(pointCloudMeshes[i]);
        }

        for (int i = 0; i < lowDetailIterationBuffers.Length; ++i) {
            lowDetailIterationBuffers[i].Release();
        }

        lowDetailMesh.GetVertexBuffer(0).Release();
        lowDetailMesh.GetIndexBuffer().Release();
        UnityEngine.Object.Destroy(lowDetailMesh);

        reductionBuffer.Release();
        pointCloudMeshes = null;
        commandBuffer.Release();
        instancedCommandBuffer.Release();
        lowDetailCommandBuffer.Release();
        voxelGrid.Release();
        occlusionGrid.Release();
        reductionDataBuffer.Release();
        finalTransformBuffer.Release();
    }

    void OnDrawGizmos() {
        // Voxel bounds cube
        Gizmos.color = Color.red;
        Gizmos.DrawWireCube(Vector3.zero, Vector3.one * voxelBounds);

        if (updateGizmoPoints) {
            Gizmos.color = Color.yellow;
            for (int i = 0; i < gizmoPoints.Count; ++i) Gizmos.DrawSphere(gizmoPoints[i], 0.025f);

            if (gizmoPoints.Count > 0) {
            Gizmos.color = Color.green;
            Gizmos.DrawSphere(newOrigin, 0.025f);

            Gizmos.DrawWireCube(newOrigin, Vector3.one * newScale);
            }
        }
    }
}
