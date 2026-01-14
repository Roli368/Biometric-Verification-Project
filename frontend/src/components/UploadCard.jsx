import React from 'react';
import { Upload, FileCheck } from 'lucide-react';

const UploadCard = ({ onUpload, selectedImage }) => {
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) onUpload(file);
  };

  return (
    <div className="relative group p-8 rounded-2xl bg-white/[0.02] border border-white/10 hover:border-blue-500/30 transition-all text-center">
      <input 
        type="file" 
        onChange={handleFileChange} 
        className="absolute inset-0 opacity-0 cursor-pointer z-10" 
      />
      {selectedImage ? (
        <div className="flex flex-col items-center gap-2 text-blue-400">
          <FileCheck size={48} className="mb-2" />
          <p className="text-sm font-bold uppercase tracking-widest">Document Scan Complete</p>
          <p className="text-xs text-slate-500">{selectedImage.name}</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-2 text-slate-400 group-hover:text-blue-400 transition-colors">
          <Upload size={48} className="mb-2" />
          <p className="text-sm font-bold uppercase tracking-widest">Upload Static Shadow</p>
          <p className="text-xs text-slate-500 font-mono">Drag & Drop Government ID</p>
        </div>
      )}
    </div>
  );
};

export default UploadCard;