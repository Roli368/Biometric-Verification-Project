import React, { useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Camera } from 'lucide-react';

const WebcamCapture = ({ onCapture, isProcessing }) => {
  const webcamRef = useRef(null);

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    onCapture(imageSrc);
  }, [webcamRef, onCapture]);

  return (
    <div className="relative rounded-2xl overflow-hidden border border-white/10 bg-black">
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        className="w-full grayscale brightness-75 hover:grayscale-0 hover:brightness-100 transition-all"
      />
      {/* Scanning Line Animation */}
      <div className="absolute inset-x-0 h-1 bg-blue-500 shadow-[0_0_15px_rgba(59,130,246,1)] animate-scan-slow pointer-events-none" />
      
      <div className="absolute bottom-6 inset-x-0 flex justify-center">
        <button 
          onClick={capture}
          disabled={isProcessing}
          className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 px-8 py-3 rounded-full font-bold flex items-center gap-2 transition active:scale-95"
        >
          <Camera size={18} />
          {isProcessing ? "Processing..." : "Verify Living Form"}
        </button>
      </div>
    </div>
  );
};

export default WebcamCapture;