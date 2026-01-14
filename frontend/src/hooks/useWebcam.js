import { useRef, useCallback } from 'react';

export const useWebcam = () => {
  const webcamRef = useRef(null);

  const captureFrame = useCallback(() => {
    if (webcamRef.current) {
      return webcamRef.current.getScreenshot();
    }
    return null;
  }, [webcamRef]);

  return {
    webcamRef,
    captureFrame
  };
};