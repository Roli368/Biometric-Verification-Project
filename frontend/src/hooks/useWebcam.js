import { useRef } from "react";

export const useWebcam = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(document.createElement("canvas"));

  const captureRGBFrame = () => {
    const video = webcamRef.current?.video;
    if (!video) return null;

    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // ✅ Optimization: willReadFrequently
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(video, 0, 0);

    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);

    let r = 0, g = 0, b = 0;
    const pixels = data.length / 4;

    for (let i = 0; i < data.length; i += 4) {
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
    }

    return [r / pixels, g / pixels, b / pixels];
  };

  return { webcamRef, captureRGBFrame };
};
