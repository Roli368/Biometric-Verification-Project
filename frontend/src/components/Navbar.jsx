import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Menu, X, Shield } from 'lucide-react';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();

  return (
    <nav className="bg-slate-950/80 backdrop-blur-md border-b border-white/5 sticky top-0 z-[100]">
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate('/')}>
          <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center font-bold text-white">Y</div>
          <span className="text-xl font-bold tracking-tighter text-white">YANTRA</span>
        </div>

        {/* Desktop Links */}
        <div className="hidden md:flex gap-8 text-xs font-mono uppercase tracking-[0.2em] text-slate-400">
          <button onClick={() => navigate('/about')} className="hover:text-white transition">Technology</button>
          <a href="/#stats" className="hover:text-white transition">Benchmarks</a>
          <button className="hover:text-white transition">Compliance</button>
        </div>

        {/* Action Button */}
        <div className="hidden md:block">
          <button 
            onClick={() => navigate('/verify')}
            className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-full text-xs font-bold uppercase tracking-widest transition shadow-lg shadow-blue-600/20"
          >
            Launch Engine
          </button>
        </div>

        {/* Hamburger */}
        <button className="md:hidden text-white" onClick={() => setIsOpen(!isOpen)}>
          {isOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>
    </nav>
  );
};

export default Navbar;