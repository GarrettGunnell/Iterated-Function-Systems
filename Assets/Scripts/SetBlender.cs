using System.Collections;
using TransformInstructions = TransformSet.TransformInstructions;
using System.Collections.Generic;
using UnityEngine;

public class SetBlender : MonoBehaviour {

    public TransformSet set1, set2;

    public AnimationCurve animationCurve;

    public bool useAnimationCurve = false;

    [Range(0.0f, 1.0f)]
    public float t = 0.0f;

    public bool animate = false;

    [Range(0.01f, 20.0f)]
    public float speed = 1.0f;
    public bool frameRateIndependent = true;

    public bool useRamp = false;

    [Range(0.01f, 20.0f)]
    public float rampSpeed = 1.0f;

    [Range(0.001f, 30.0f)]
    public float epsilon = 1.0f;
    
    public TransformInstructions finalTransform = new TransformInstructions();

    private List<TransformInstructions> blendedSet = new List<TransformInstructions>();

    private bool swap = false;

    public List<TransformInstructions> GetBlendedSet() {
        return blendedSet;
    }

    public TransformInstructions GetFinalTransform() {
        return finalTransform;
    }

    TransformInstructions InterpolateInstructions(TransformInstructions t1, TransformInstructions t2, float t) {
        TransformInstructions interpolatedInstructions = new TransformInstructions();

        interpolatedInstructions.scale = Vector3.LerpUnclamped(t1.scale, t2.scale, t);
        interpolatedInstructions.shearX = Vector3.LerpUnclamped(t1.shearX, t2.shearX, t);
        interpolatedInstructions.shearY = Vector3.LerpUnclamped(t1.shearY, t2.shearY, t);
        interpolatedInstructions.shearZ = Vector3.LerpUnclamped(t1.shearZ, t2.shearZ, t);
        interpolatedInstructions.translate = Vector3.LerpUnclamped(t1.translate, t2.translate, t);

        Quaternion r1 = Quaternion.Euler(t1.rotate);
        Quaternion r2 = Quaternion.Euler(t2.rotate);
        Quaternion r3 = Quaternion.SlerpUnclamped(r1, r2, t);

        interpolatedInstructions.rotate = r3.eulerAngles;

        return interpolatedInstructions;
    }


    private void BlendSets() {
        blendedSet.Clear();

        // Copy instruction sets so the following modifications don't ruin the originals
        List<TransformInstructions> instructionSet1 = new List<TransformInstructions>(set1.transformSet);
        List<TransformInstructions> instructionSet2 = new List<TransformInstructions>(set2.transformSet);

        // Apply post transform to all instruction sets
        for (int i = 0; i < instructionSet1.Count; ++i) {
            instructionSet1[i] += set1.postTransform;
        }

        for (int i = 0; i < instructionSet2.Count; ++i) {
            instructionSet2[i] += set2.postTransform;
        }

        // In order to blend smaller instruction sets with larger instruction sets, append identity matrices to smaller set
        int sizeDifference = Mathf.Abs(instructionSet1.Count - instructionSet2.Count);
        if (instructionSet1.Count < instructionSet2.Count) {
            for (int i = 0; i < sizeDifference; ++i) {
                instructionSet1.Add(TransformSet.GetIdentity());
            }
        } else if (instructionSet2.Count < instructionSet1.Count) {
            for (int i = 0; i < sizeDifference; ++i) {
                instructionSet2.Add(TransformSet.GetIdentity());
            }
        }

        // Blend instruction sets and create list of affine transformations
        for (int i = 0; i < instructionSet1.Count; ++i) {
            float blendFactor = animationCurve.Evaluate(t); // TO DO: Easing Functions

            if (swap)
                blendedSet.Add(InterpolateInstructions(instructionSet2[i], instructionSet1[i], blendFactor));
            else
                blendedSet.Add(InterpolateInstructions(instructionSet1[i], instructionSet2[i], blendFactor));
        }
    }

    private List<TransformInstructions> moveTowardSet = new List<TransformInstructions>();

    Vector3 ExpDecay(Vector3 a, Vector3 b, float decay) {
        return Vector3.LerpUnclamped(a, b, decay);
    }

    float ramp = 0;
    TransformInstructions MoveTowardInstructions(TransformInstructions t1, TransformInstructions t2, float decay) {
        TransformInstructions i = new TransformInstructions();

        float decayDeltaTime = 0.0f;

        if (frameRateIndependent) decayDeltaTime = decay * Time.deltaTime;
        else decayDeltaTime = decay;

        if (useRamp) decayDeltaTime *= animationCurve.Evaluate(ramp);

        decayDeltaTime = Mathf.Min(decayDeltaTime, 1.0f);

        i.scale = ExpDecay(t1.scale, t2.scale, decayDeltaTime);
        i.shearX = ExpDecay(t1.shearX, t2.shearX, decayDeltaTime);
        i.shearY = ExpDecay(t1.shearY, t2.shearY, decayDeltaTime);
        i.shearZ = ExpDecay(t1.shearZ, t2.shearZ, decayDeltaTime);
        i.translate = ExpDecay(t1.translate, t2.translate, decayDeltaTime);

        Quaternion r1 = Quaternion.Euler(t1.rotate);
        Quaternion r2 = Quaternion.Euler(t2.rotate);
        Quaternion r3 = Quaternion.Slerp(r1, r2, Mathf.Min(decayDeltaTime, 1.0f));

        i.rotate = r3.eulerAngles;

        return i;
    }

    private void MoveTowardSet() {
        blendedSet.Clear();

        if (moveTowardSet.Count != set1.transformSet.Count) moveTowardSet = new List<TransformInstructions>(set1.transformSet);

        for (int i = 0; i < moveTowardSet.Count; ++i) {
            moveTowardSet[i] = MoveTowardInstructions(moveTowardSet[i], set2.transformSet[i], speed);
            blendedSet.Add(moveTowardSet[i]);
        }
    }

    private void OnEnable() {
        t = 0;

        moveTowardSet = new List<TransformInstructions>(set1.transformSet);

        if (useAnimationCurve)
            BlendSets();
        else
            MoveTowardSet();
    }

    private bool paused = false;
    private void Update() {

        if (Input.GetKeyDown("space")) animate = !animate;

        // Update ramp timer for lerp smoothing
        if (frameRateIndependent) {
            ramp += Time.deltaTime * rampSpeed;
        } else {
            ramp += rampSpeed;
        }

        // Automate T
        if (animate) {
            if (frameRateIndependent) {
                t += Time.deltaTime * speed;
            } else {
                t += speed;
            }

            if (useAnimationCurve) {
                t = Mathf.Clamp(t, 0.0f, 1.0f);
                if (t >= 1) {
                    swap = !swap;
                    t = 0;

                    if (swap) set1.ApplyPreset();
                    else set2.ApplyPreset();
                }
            } else {
                if (t >= epsilon) {
                    t = 0;
                    ramp = 0;
                    set2.ApplyPreset();
                }
            }
        }

        // Update output set
        if (useAnimationCurve) {
            BlendSets();
        } else {
            if (Input.GetKeyDown("f")) {
                ramp = 0;
                set2.ApplyPreset();
            }
            MoveTowardSet();
        }
    }
}
