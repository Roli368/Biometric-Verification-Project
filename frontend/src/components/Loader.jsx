import React from 'react';

const Loader = ({ text = "Processing..." }) => (
  <div className="flex flex-col items-center justify-center space-y-6 h-full py-20">
    <div className="relative w-20 h-20">
      <div className="absolute inset-0 border-2 border-blue-500/20 rounded-full"></div>
      <div className="absolute inset-0 border-t-2 border-blue-500 rounded-full animate-spin"></div>
      <div className="absolute inset-4 border border-blue-400/30 rounded-full animate-pulse"></div>
    </div>
    <p className="text-blue-500 font-mono text-xs animate-pulse tracking-[0.3em] uppercase">{text}</p>
  </div>
);

export default Loader;