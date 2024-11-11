using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChaosGame : IteratedFunctionSystem {
    [Range(1, 10)]
    public int iterationsPerFrame = 1;

    public override void IterateSystem() {
        for (int iteration = 0; iteration < iterationsPerFrame; ++iteration) {
            for (int i = 0; i < (int)batchCount; ++i) {
                particleUpdater.SetInt("_TransformationCount", affineTransformations.GetTransformCount());
                particleUpdater.SetInt("_ParticleCount", (int)particlesPerBatch);
                particleUpdater.SetInt("_Seed", Mathf.CeilToInt(Random.Range(1, 1000000)));
                particleUpdater.SetBuffer(1, "_VertexBuffer", pointCloudMeshes[i].GetVertexBuffer(0));
                particleUpdater.SetBuffer(1, "_Transformations", affineTransformations.GetAffineBuffer());
                particleUpdater.Dispatch(1, Mathf.CeilToInt(particlesPerBatch / threadsPerGroup), 1, 1);
            }
        }
    }
}
