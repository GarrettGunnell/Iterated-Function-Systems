Shader "Custom/InstancedParticle" {

	SubShader {

		Pass {
			ZWrite On

			Tags {
				"RenderType" = "Opaque"
				"LightMode" = "ForwardBase"
			}

			CGPROGRAM

			#pragma vertex vp
			#pragma fragment fp

			#include "UnityCG.cginc"
			#define UNITY_INDIRECT_DRAW_ARGS IndirectDrawIndexedArgs
			#include "UnityIndirect.cginc"
			#include "UnityPBSLighting.cginc"
            #include "AutoLight.cginc"

			struct VertexData {
				float4 vertex : POSITION;
			};

			struct v2f {
				float4 pos : SV_POSITION;
				float occlusion : TEXCOORD0;
				float3 worldPos : TEXCOORD1;
				float outOfBounds : TEXCOORD2;
			};

			
			StructuredBuffer<float> _OcclusionGrid;
			int _GridSize, _GridBounds;

			uint to1D(uint3 pos) {
				return pos.x + pos.y * _GridSize + pos.z * _GridSize * _GridSize;
			}

			float getTrilinearVoxel(float3 pos) {
				float v = 0;

				float boundsExtent = _GridBounds;

				if (abs(dot(pos, float3(1, 0, 0))) <= boundsExtent &&
					abs(dot(pos, float3(0, 1, 0))) <= boundsExtent &&
					abs(dot(pos, float3(0, 0, 1))) <= boundsExtent)
				{
					float3 seedPos = pos;
					seedPos += (_GridBounds / 2.0f);
					seedPos /= _GridBounds;
					seedPos *= _GridSize;
					// seedPos -= 0.5f;

					uint3 vi = floor(seedPos);

					float weight1 = 0.0f;
					float weight2 = 0.0f;
					float weight3 = 0.0f;
					float value = 0.0f;

					for (int i = 0; i < 2; ++i) {
						weight1 = 1 - min(abs(seedPos.x - (vi.x + i)), _GridSize);
						for (int j = 0; j < 2; ++j) {
							weight2 = 1 - min(abs(seedPos.y - (vi.y + j)), _GridSize);
							for (int k = 0; k < 2; ++k) {
								weight3 = 1 - min(abs(seedPos.z - (vi.z + k)), _GridSize);
								value += weight1 * weight2 * weight3 * _OcclusionGrid[to1D(vi + uint3(i, j, k))];
							}
						}
					}

					v = value;
				}

				return v;
			}

			float4x4 _FinalTransform;
			float _OcclusionMultiplier, _OcclusionAttenuation;
			float3 _ParticleColor, _OcclusionColor;
			
			StructuredBuffer<float4x4> _Transformations;

			v2f vp(VertexData v, uint svInstanceID : SV_INSTANCEID) {
				InitIndirectDrawArgs(0);

				v2f i;
				
				uint instanceID = GetIndirectInstanceID(svInstanceID);

				// float3 pos = v.vertex.xyz;
				// pos += (_GridBounds / 2.0f);
				// pos /= _GridBounds;
				// pos *= _GridSize;

				// if (any(uint3(pos) > _GridSize) || any(pos < 0)) return;

				float4 pos = mul(_FinalTransform, mul(_Transformations[instanceID], v.vertex));
				// float4 pos = mul(_FinalTransform, v.vertex);

				float gridBounds = _GridBounds * 0.5f;

				i.outOfBounds = 0.0f;
				if (any(pos > gridBounds) || any(pos < -gridBounds)) i.outOfBounds = 1.0f;

				i.pos = UnityObjectToClipPos(pos);
				// i.occlusion = _OcclusionGrid[to1D(pos)];
				i.occlusion = 0;
				i.worldPos = pos;
				return i;
			}

			float hash(uint n) {
				// integer hash copied from Hugo Elias
				n = (n << 13U) ^ n;
				n = n * (n * n * 15731U + 0x789221U) + 0x1376312589U;
				return float(n & uint(0x7fffffffU)) / float(0x7fffffff);
			}

			float4 fp(v2f i) : SV_TARGET {
				float3 col = _ParticleColor;
				// return 1;
				clip(-i.outOfBounds);
				
				float occlusion = getTrilinearVoxel(i.worldPos);

				occlusion = pow(saturate(occlusion * _OcclusionMultiplier), _OcclusionAttenuation);
				// return 1;
				return float4(lerp(_OcclusionColor, col, occlusion), 1);
			}

			ENDCG
		}
	}
}