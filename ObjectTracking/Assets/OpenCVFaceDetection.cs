using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

public class OpenCVFaceDetection : MonoBehaviour
{
    // Define the functions which can be called from the .dll.
    internal static class OpenCVInterop
    {
        [DllImport("OpenCV")]
        internal static extern int Init(ref int outCameraWidth, ref int outCameraHeight);

        [DllImport("OpenCV")]
        internal static extern int Close();

        [DllImport("OpenCV")]
        internal static extern int SetScale(int downscale);

        [DllImport("OpenCV")]
        internal unsafe static extern void Detect(CvCircle* outFaces, int maxOutFacesCount, ref int outDetectedFacesCount);

        [DllImport("OpenCV")]
        internal unsafe static extern int Track(CvRectangle* outTracking, int maxTrackingCount, ref int outTrackingsCount);
    }

    [StructLayout(LayoutKind.Sequential, Size = 12)]
    public struct CvCircle
    {
        public int X, Y, Radius;
    }

    public struct CvRectangle
    {
        public int Width, Height, X, Y;
    }


    // Start is called before the first frame update
    public static List<Vector2> NormalizedFacePositions { get; private set; }
    public static List<Vector2> NormalizedTrackingPositions { get; private set; }
    public static Vector2 CameraResolution;

    /// <summary>
    /// Downscale factor to speed up detection.
    /// </summary>
    private const int DetectionDownScale = 1;

    private bool _ready;
    private int _maxFaceDetectCount = 5;
    private int _maxTrackingCount = 5;
    private CvCircle[] _faces;
    private CvRectangle[] _tracking;

    void Start()
    {
        int camWidth = 0, camHeight = 0;
        int result = OpenCVInterop.Init(ref camWidth, ref camHeight);
        if (result < 0)
        {
            if (result == -1)
            {
                Debug.LogWarningFormat("[{0}] Failed to find cascades definition.", GetType());
            }
            else if (result == -2)
            {
                Debug.LogWarningFormat("[{0}] Failed to open camera stream.", GetType());
            }

            return;
        }

        CameraResolution = new Vector2(camWidth, camHeight);
        _faces = new CvCircle[_maxFaceDetectCount];
        _tracking = new CvRectangle[_maxTrackingCount];
        NormalizedFacePositions = new List<Vector2>();
        NormalizedTrackingPositions = new List<Vector2>();
        OpenCVInterop.SetScale(DetectionDownScale);
        _ready = true;
    }

    void OnApplicationQuit()
    {
        if (_ready)
        {
            OpenCVInterop.Close();
        }
    }

    void Update()
    {
        if (!_ready) {
            return;
        }
        ObjectTracking();

    }

    void FaceDetection()
    {
        int detectedFaceCount = 0;
        unsafe
        {
            fixed (CvCircle* outFaces = _faces)
            {
                OpenCVInterop.Detect(outFaces, _maxFaceDetectCount, ref detectedFaceCount);
            }
        }

        NormalizedFacePositions.Clear();
        for (int i = 0; i < detectedFaceCount; i++)
        {
            NormalizedFacePositions.Add(new Vector2((_faces[i].X * DetectionDownScale) / CameraResolution.x, 1f - ((_faces[i].Y * DetectionDownScale) / CameraResolution.y)));
        }
    }

    void ObjectTracking()
    {
        int detectedTrackingCount = 0;
        unsafe
        {
            fixed (CvRectangle* outTracking = _tracking)
            {
                OpenCVInterop.Track(outTracking, _maxTrackingCount, ref detectedTrackingCount);
            }
        }

        NormalizedFacePositions.Clear();
        for (int i = 0; i < detectedTrackingCount; i++)
        {
            NormalizedTrackingPositions.Add(new Vector2((_tracking[i].X * DetectionDownScale) / CameraResolution.x, 1f - ((_tracking[i].Y * DetectionDownScale) / CameraResolution.y)));
        }
    }
}

