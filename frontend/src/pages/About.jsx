import React from 'react';

const About = () => {
  const aspects = [
    { title: "Swarupa (Identity)", desc: "Siamese Vision Transformers (ViT) reconcile domain shift between grainy ID photos and 4K selfies.", tech: "ArcFace Loss + ViT Backbone" },
    { title: "Prana (Life)", desc: " physiological signals invisible to the eye prove biological existence via blood flow rhythm.", tech: "Remote PPG (rPPG)" },
    { title: "Dharma (Fairness)", desc: "Mitigating demographic bias differentially across race, gender, and age for inclusive security.", tech: "NIST FRVT Compliance" }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-white p-20">
      <div className="max-w-4xl mx-auto space-y-16">
        <header>
          <h1 className="text-4xl font-extrabold mb-4 tracking-tight">The Biometric Yantra</h1>
          <p className="text-slate-400 text-lg">Engineering an impartial judge for the digital age.</p>
        </header>

        <div className="grid gap-12">
          {aspects.map((aspect, i) => (
            <div key={i} className="border-l border-white/10 pl-8 relative">
              <div className="absolute -left-[1.5px] top-0 h-8 w-1 bg-blue-600"></div>
              <h2 className="text-xl font-bold mb-3">{aspect.title}</h2>
              <p className="text-slate-400 leading-relaxed mb-4">{aspect.desc}</p>
              <span className="text-[10px] font-mono text-blue-500 bg-blue-500/10 px-2 py-1 rounded">
                CORE TECH: {aspect.tech}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default About;