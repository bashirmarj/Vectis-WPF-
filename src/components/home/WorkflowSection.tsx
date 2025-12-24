import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileText, Cog, CheckCircle, Truck } from 'lucide-react';

const steps = [
  { id: "01", title: "Upload", text: "Submit your CAD files securely to our portal for instant parsing.", icon: Upload },
  { id: "02", title: "Quote", text: "Get detailed pricing and lead time within 24 hours.", icon: FileText },
  { id: "03", title: "Manufacture", text: "Production begins using high-speed 5-axis CNC centers.", icon: Cog },
  { id: "04", title: "Quality", text: "Rigorous inspection ensures every part meets specifications.", icon: CheckCircle },
  { id: "05", title: "Deliver", text: "Parts are packed and shipped directly to your facility.", icon: Truck },
];

const WorkflowSection: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % steps.length);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <section className="relative z-10 py-24 bg-transparent">
      <div className="container mx-auto px-6">
        {/* Header */}
        <div className="mb-16 text-center max-w-2xl mx-auto">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-3xl md:text-4xl font-bold text-slate-900 mb-4"
          >
            Streamlined Workflow
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-slate-900 font-light text-lg"
          >
            From digital file to physical part in record time.
          </motion.p>
        </div>

        {/* Desktop Layout */}
        <div className="hidden md:block">
          {/* Circles with Connection Line */}
          <div className="relative flex items-center justify-between max-w-4xl mx-auto mb-12">
            {/* Background Connection Line */}
            <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-slate-200 -translate-y-1/2 z-0" />
            
            {/* Animated Progress Line */}
            <motion.div 
              className="absolute top-1/2 left-0 h-0.5 bg-primary -translate-y-1/2 z-0"
              initial={{ width: '0%' }}
              animate={{ width: `${(activeStep / (steps.length - 1)) * 100}%` }}
              transition={{ duration: 0.5, ease: 'easeInOut' }}
            />

            {/* Step Circles */}
            {steps.map((step, index) => {
              const isActive = index === activeStep;
              const isCompleted = index < activeStep;
              const Icon = step.icon;

              return (
                <motion.div
                  key={step.id}
                  initial={{ opacity: 0, scale: 0 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  className="relative z-10 flex flex-col items-center"
                >
                  {/* Circle */}
                  <motion.div
                    className={`
                      relative w-20 h-20 rounded-full flex items-center justify-center cursor-pointer
                      transition-colors duration-300
                      ${isActive 
                        ? 'bg-primary text-white' 
                        : isCompleted 
                          ? 'bg-primary/20 text-primary border-2 border-primary' 
                          : 'bg-white text-slate-500 border-2 border-slate-200'
                      }
                    `}
                    animate={{
                      scale: isActive ? 1.15 : 1,
                      boxShadow: isActive 
                        ? '0 0 30px rgba(var(--primary-rgb, 59, 130, 246), 0.5)' 
                        : '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                    }}
                    transition={{ duration: 0.3 }}
                    onClick={() => setActiveStep(index)}
                  >
                    <Icon className="w-8 h-8" />
                    
                    {/* Active Ring Animation */}
                    {isActive && (
                      <motion.div
                        className="absolute inset-0 rounded-full border-2 border-primary"
                        initial={{ scale: 1, opacity: 1 }}
                        animate={{ scale: 1.4, opacity: 0 }}
                        transition={{ 
                          duration: 1.5, 
                          repeat: Infinity,
                          ease: 'easeOut'
                        }}
                      />
                    )}
                  </motion.div>

                  {/* Step Number */}
                  <motion.span 
                    className={`
                      mt-4 text-sm font-semibold tracking-wider
                      ${isActive ? 'text-primary' : 'text-slate-400'}
                    `}
                    animate={{ scale: isActive ? 1.1 : 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    {step.id}
                  </motion.span>
                </motion.div>
              );
            })}
          </div>

          {/* Active Step Details */}
          <motion.div 
            key={activeStep}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="text-center max-w-lg mx-auto"
          >
            <h3 className="text-2xl font-bold text-slate-900 mb-3 uppercase tracking-wide">
              {steps[activeStep].title}
            </h3>
            <p className="text-slate-600 leading-relaxed">
              {steps[activeStep].text}
            </p>
          </motion.div>
        </div>

        {/* Mobile Layout - Vertical Timeline */}
        <div className="md:hidden space-y-0">
          {steps.map((step, index) => {
            const isActive = index === activeStep;
            const isCompleted = index < activeStep;
            const Icon = step.icon;
            const isLast = index === steps.length - 1;

            return (
              <motion.div
                key={step.id}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                className="relative flex gap-4"
              >
                {/* Left side - Circle and Line */}
                <div className="flex flex-col items-center">
                  {/* Circle */}
                  <motion.div
                    className={`
                      relative w-14 h-14 rounded-full flex items-center justify-center flex-shrink-0
                      transition-colors duration-300
                      ${isActive 
                        ? 'bg-primary text-white' 
                        : isCompleted 
                          ? 'bg-primary/20 text-primary border-2 border-primary' 
                          : 'bg-white text-slate-500 border-2 border-slate-200'
                      }
                    `}
                    animate={{
                      scale: isActive ? 1.1 : 1,
                      boxShadow: isActive 
                        ? '0 0 20px rgba(var(--primary-rgb, 59, 130, 246), 0.4)' 
                        : '0 2px 4px rgba(0, 0, 0, 0.1)',
                    }}
                    transition={{ duration: 0.3 }}
                    onClick={() => setActiveStep(index)}
                  >
                    <Icon className="w-6 h-6" />
                    
                    {isActive && (
                      <motion.div
                        className="absolute inset-0 rounded-full border-2 border-primary"
                        initial={{ scale: 1, opacity: 1 }}
                        animate={{ scale: 1.3, opacity: 0 }}
                        transition={{ 
                          duration: 1.5, 
                          repeat: Infinity,
                          ease: 'easeOut'
                        }}
                      />
                    )}
                  </motion.div>

                  {/* Connecting Line */}
                  {!isLast && (
                    <div className="relative w-0.5 h-16 bg-slate-200">
                      <motion.div
                        className="absolute top-0 left-0 w-full bg-primary"
                        initial={{ height: '0%' }}
                        animate={{ height: isCompleted || isActive ? '100%' : '0%' }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                  )}
                </div>

                {/* Right side - Content */}
                <div className={`pb-8 ${isLast ? 'pb-0' : ''}`}>
                  <motion.span 
                    className={`
                      text-xs font-semibold tracking-wider
                      ${isActive ? 'text-primary' : 'text-slate-400'}
                    `}
                  >
                    STEP {step.id}
                  </motion.span>
                  <h3 className={`
                    text-lg font-bold mb-1 uppercase tracking-wide transition-colors duration-300
                    ${isActive ? 'text-slate-900' : 'text-slate-700'}
                  `}>
                    {step.title}
                  </h3>
                  <p className={`
                    text-sm leading-relaxed transition-colors duration-300
                    ${isActive ? 'text-slate-600' : 'text-slate-500'}
                  `}>
                    {step.text}
                  </p>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Step Indicators - Desktop */}
        <div className="hidden md:flex justify-center gap-2 mt-8">
          {steps.map((_, index) => (
            <button
              key={index}
              onClick={() => setActiveStep(index)}
              className={`
                w-2 h-2 rounded-full transition-all duration-300
                ${index === activeStep 
                  ? 'bg-primary w-6' 
                  : 'bg-slate-300 hover:bg-slate-400'
                }
              `}
              aria-label={`Go to step ${index + 1}`}
            />
          ))}
        </div>
      </div>
    </section>
  );
};

export default WorkflowSection;
