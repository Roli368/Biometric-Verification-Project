import React from "react";
import { Link } from "react-router-dom";
import { Shield, Database, Terminal, Cpu } from "lucide-react";

export default function Navbar() {
  return (
    <header className="sticky top-0 w-full z-[100] bg-[#020617] border-b border-white/5">
      {/* TOP UTILITY TIER */}
      <div className="h-9 bg-black/40 flex items-center px-6 justify-between text-[10px] font-mono tracking-widest text-slate-500 border-b border-white/5">
        <div className="flex gap-6">
          <span className="flex items-center gap-2">
            <div className="w-1 h-1 bg-cyan-500 rounded-full animate-pulse"/> 
            SYSTEM_REPLICA: ACTIVE
          </span>
          <span className="hidden md:block">LATENCY: 14ms</span>
        </div>
        <div className="flex gap-4">
          <span className="text-cyan-500/50 underline">SECURITY_AUDIT_PASS</span>
        </div>
      </div>

      {/* PRIMARY NAV TIER */}
      <nav className="h-20 px-6 md:px-10 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <Link to="/" className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-600 flex items-center justify-center text-white font-black text-xl rounded-sm">Y</div>
            <div className="flex flex-col">
              <span className="text-xl font-black tracking-tighter text-white uppercase italic">Yantra</span>
              <span className="text-[9px] font-mono text-cyan-500 tracking-[0.3em]">NEURAL_ID</span>
            </div>
          </Link>
          
          <div className="hidden lg:flex gap-8 text-[11px] font-bold tracking-widest text-slate-400">
            <Link to="/" className="hover:text-cyan-400 transition-colors">SYSTEMS</Link>
            <Link to="/verify" className="hover:text-cyan-400 transition-colors">VERIFY_ENGINE</Link>
            <Link to="/about" className="hover:text-cyan-400 transition-colors">RESEARCH</Link>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button className="hidden sm:flex items-center gap-2 text-slate-400 hover:text-white font-mono text-[10px] border border-white/10 px-4 py-2">
            <Terminal size={14}/> CONSOLE
          </button>
          <Link to="/verify" className="bg-blue-600 hover:bg-cyan-500 text-white hover:text-slate-950 px-6 py-2.5 rounded-full text-xs font-black tracking-widest transition-all">
            RUN DEMO
          </Link>
        </div>
      </nav>
    </header>
  );
}