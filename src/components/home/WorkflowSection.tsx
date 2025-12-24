import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileText, Cog, CheckCircle, Truck, Check } from 'lucide-react';

const steps = [
  { id: "01", title: "Upload", text: "Submit your CAD files securely to our portal for instant parsing.", icon: Upload, duration: 2600 },
  { id: "02", title: "Quote", text: "Get detailed pricing and lead time within 24 hours.", icon: FileText, duration: 2250 },
  { id: "03", title: "Manufacture", text: "Production begins using high-speed 5-axis CNC centers.", icon: Cog, duration: 3400 },
  { id: "04", title: "Quality", text: "Rigorous inspection ensures every part meets specifications.", icon: CheckCircle, duration: 2250 },
  { id: "05", title: "Deliver", text: "Parts are packed and shipped directly to your facility.", icon: Truck, duration: 2600 },
];

const WorkflowSection: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);

  useEffect(() => {
    let timeout: ReturnType<typeof setTimeout>;

    const runSequence = () => {
      const currentDuration = steps[activeStep].duration;
      
      timeout = setTimeout(() => {
        setCompletedSteps(prev => {
           if (activeStep === steps.length - 1) return [];
           if (!prev.includes(activeStep)) return [...prev, activeStep];
           return prev;
        });
        setActiveStep(prev => (prev + 1) % steps.length);
      }, currentDuration);
    };

    runSequence();

    return () => clearTimeout(timeout);
  }, [activeStep]);

  return (
    <section className="relative z-10 py-32 bg-transparent border-t border-slate-200/60">
      <div className="container mx-auto px-6">
        
        {/* Header */}
        <div className="mb-16">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-1 bg-primary rounded-full"></div>
            <span className="text-sm font-mono tracking-[0.2em] text-slate-900 uppercase">Workflow</span>
          </div>
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-slate-900">Streamlined Workflow</h2>
          <p className="text-slate-900 text-lg mt-4 max-w-2xl">
            From digital file to physical part in record time.
          </p>
        </div>

        {/* Timeline Container */}
        <div className="relative max-w-6xl mx-auto">
          
          {/* Background Rail (Desktop) */}
          <div className="absolute top-[71px] left-0 w-full h-[2px] bg-slate-200 hidden md:block" />
          
          {/* Active Progress Rail */}
          <div 
            className="absolute top-[71px] left-0 h-[2px] bg-emerald-500 hidden md:block transition-all duration-700 ease-linear shadow-[0_0_10px_rgba(16,185,129,0.5)]"
            style={{ 
              width: `${(completedSteps.length / (steps.length - 1)) * 100}%`,
              opacity: completedSteps.length > 0 ? 1 : 0
            }} 
          />

          <div className="grid grid-cols-1 md:grid-cols-5 gap-8 relative z-10">
            {steps.map((step, index) => {
              const isActive = index === activeStep;
              const isComplete = completedSteps.includes(index);
              
              return (
                <div key={step.id} className="relative flex flex-col items-center text-center group">
                  
                  {/* --- LARGE CIRCLE NODE --- */}
                  <div className={`
                      relative w-36 h-36 rounded-full flex flex-col items-center justify-center mb-6 transition-all duration-500 overflow-hidden
                      bg-white shadow-xl
                      ${isActive ? 'scale-110 ring-4 ring-emerald-500/20 z-20' : 'scale-100 z-10'}
                      ${isComplete ? 'border-4 border-emerald-500' : 'border-4 border-slate-100'}
                  `}>
                      
                      {/* 1. Progressive Shade Animation (Background Layer) */}
                      {isActive && (
                        <motion.div 
                          className="absolute inset-0 bg-emerald-100"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ duration: step.duration / 1000, ease: "linear" }}
                        />
                      )}
                      
                      {/* 2. Completed State Fill */}
                      {isComplete && !isActive && (
                         <div className="absolute inset-0 bg-emerald-50" />
                      )}

                      {/* 3. Content Inside Circle */}
                      <div className="relative z-20 flex flex-col items-center gap-1">
                        {isComplete && !isActive ? (
                            <div className="w-8 h-8 bg-emerald-500 rounded-full flex items-center justify-center shadow-sm mb-1">
                                <Check size={16} className="text-white" strokeWidth={3} />
                            </div>
                        ) : (
                            <step.icon 
                                size={28} 
                                strokeWidth={1.5}
                                className={`mb-1 transition-colors duration-300 ${isActive ? 'text-emerald-600' : 'text-slate-400'}`} 
                            />
                        )}
                        
                        <h3 className={`text-xs font-bold uppercase tracking-wider ${isActive || isComplete ? 'text-emerald-900' : 'text-slate-400'}`}>
                            {step.title}
                        </h3>
                        <div className={`text-[10px] font-mono ${isActive || isComplete ? 'text-emerald-700/70' : 'text-slate-300'}`}>
                            STEP {step.id}
                        </div>
                      </div>

                      {/* 4. Active Pulse Ring (Inner Border) */}
                      {isActive && (
                          <div className="absolute inset-0 rounded-full border border-emerald-500 animate-pulse opacity-50" />
                      )}
                  </div>

                  {/* --- DESCRIPTION TEXT (Below) --- */}
                  <div className={`transition-all duration-500 max-w-[220px] px-2 ${isActive || isComplete ? 'opacity-100 translate-y-0' : 'opacity-40 translate-y-2'}`}>
                    <p className="text-sm text-slate-900 font-medium leading-relaxed">
                        {step.text}
                    </p>
                  </div>

                  {/* Mobile Connecting Line */}
                  {index < steps.length - 1 && (
                      <div className="absolute h-16 w-[2px] bg-slate-200 -bottom-14 left-1/2 -translate-x-1/2 md:hidden" />
                  )}

                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};

export default WorkflowSection;
