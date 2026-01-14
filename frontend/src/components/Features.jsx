import React from 'react';
import { UserCheck, Zap, ShieldCheck } from 'lucide-react';

const FeatureCard = ({ icon: Icon, title, desc, tech }) => (
  <div className="p-8 rounded-2xl bg-white/[0.03] border border-white/5 hover:border-blue-500/30 transition-all group">
    <div className="w-12 h-12 rounded-lg bg-white/5 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
      <Icon className="text-blue-500" />
    </div>
    <h3 className="text-xl font-bold mb-3 text-white">{title}</h3>
    <p className="text-slate-400 text-sm leading-relaxed mb-4">{desc}</p>
    <span className="text-[10px] font-mono text-blue-400/60 uppercase tracking-widest">{tech}</span>
  </div>
);

const Features = () => {
  return (
    <section id="features" className="py-24 px-6 max-w-6xl mx-auto">
      <div className="grid md:grid-cols-3 gap-8">
        <FeatureCard 
          icon={UserCheck}
          title="Swarupa Matching"
          tech="Vision Transformer"
          desc="Matching static ID photos against live subjects with high-dimensional feature vectors."
        />
        <FeatureCard 
          icon={Zap}
          title="Prana Analysis"
          tech="Remote PPG"
          desc="Detecting biological life signals through micro-variations in skin reflectance."
        />
        <FeatureCard 
          icon={ShieldCheck}
          title="Dharma Audit"
          tech="Fairness Engine"
          desc="Ensuring demographic parity and mitigating algorithmic bias across all user groups."
        />
      </div>
    </section>
  );
};

export default Features;