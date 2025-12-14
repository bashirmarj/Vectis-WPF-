import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { useRef, useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import AnimatedSection from "@/components/home/AnimatedSection";
import AnimatedCounter from "@/components/home/AnimatedCounter";
import ProcessSectionGemini from "@/components/home/ProcessSectionGemini";
import ScrollIndicator from "@/components/home/ScrollIndicator";
import ParticleBackground from "@/components/home/ParticleBackground";
import cncMachiningImg from "@/assets/cnc-machining-showcase.png";
import sheetMetalImg from "@/assets/sheet-metal-new.png";
import heatTreatmentImg from "@/assets/heat-treatment-new.png";
import dieCastingImg from "@/assets/die-casting.png";
import wireEdmImg from "@/assets/wire-edm.png";

const Index = () => {
  const services = [
    {
      title: "PROTOTYPE DESIGN",
      description: "From concept to functional prototype. Fast turnaround, precision engineering.",
    },
    {
      title: "CUSTOM PARTS",
      description: "High-quality components manufactured to exact specifications.",
    },
  ];

  const stats = [
    { value: "25+", label: "Years Experience" },
    { value: "500+", label: "Projects Completed" },
    { value: "98%", label: "On-Time Delivery" },
    { value: "ISO", label: "9001 Certified" },
  ];

  const capabilities = [
    {
      title: "CNC MACHINING",
      description: "5-axis precision milling and turning for complex geometries.",
      image: cncMachiningImg,
    },
    {
      title: "WIRE EDM",
      description: "Complex geometries with tight tolerances in hardened materials.",
      image: wireEdmImg,
      imageStyle: "bg-[length:70%] bg-center",
    },
    {
      title: "SHEET METAL",
      description: "Laser cutting, bending, and forming at any scale.",
      image: sheetMetalImg,
    },
    {
      title: "HEAT TREATMENT",
      description: "Surface Hardening, PVD/CVD Coating, Chemical and Vacuum Heat Treatment",
      image: heatTreatmentImg,
    },
    {
      title: "DIE CASTING",
      description: "Precision die casting service for customized metal parts",
      image: dieCastingImg,
    },
  ];

  // Drag to scroll state
  const marqueeRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [startX, setStartX] = useState(0);
  const dragStartPositionRef = useRef(0);
  const resumeTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Animation refs
  const positionRef = useRef(0);
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

  const TOTAL_MARQUEE_WIDTH = 2410;
  const ANIMATION_DURATION = 35;
  const SPEED = TOTAL_MARQUEE_WIDTH / (ANIMATION_DURATION * 1000);

  const animate = useCallback((currentTime: number) => {
    if (!containerRef.current) return;

    if (lastTimeRef.current === 0) {
      lastTimeRef.current = currentTime;
    }

    const deltaTime = currentTime - lastTimeRef.current;
    lastTimeRef.current = currentTime;

    positionRef.current -= SPEED * deltaTime;

    if (positionRef.current <= -TOTAL_MARQUEE_WIDTH) {
      positionRef.current += TOTAL_MARQUEE_WIDTH;
    }

    containerRef.current.style.transform = `translateX(${positionRef.current}px)`;
    animationRef.current = requestAnimationFrame(animate);
  }, [SPEED, TOTAL_MARQUEE_WIDTH]);

  const startAnimation = useCallback(() => {
    lastTimeRef.current = 0;
    animationRef.current = requestAnimationFrame(animate);
  }, [animate]);

  const stopAnimation = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  }, []);

  useEffect(() => {
    startAnimation();
    return () => stopAnimation();
  }, [startAnimation, stopAnimation]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!marqueeRef.current) return;

    if (resumeTimeoutRef.current) {
      clearTimeout(resumeTimeoutRef.current);
      resumeTimeoutRef.current = null;
    }

    stopAnimation();
    setIsDragging(true);
    setStartX(e.pageX);
    dragStartPositionRef.current = positionRef.current;
    marqueeRef.current.style.cursor = "grabbing";
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    if (marqueeRef.current) {
      marqueeRef.current.style.cursor = "grab";
    }

    positionRef.current = positionRef.current % TOTAL_MARQUEE_WIDTH;
    if (positionRef.current > 0) {
      positionRef.current -= TOTAL_MARQUEE_WIDTH;
    }

    resumeTimeoutRef.current = setTimeout(() => {
      startAnimation();
    }, 2000);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !containerRef.current) return;
    e.preventDefault();
    
    const dragDelta = (e.pageX - startX) * 0.8;
    const newPosition = dragStartPositionRef.current + dragDelta;
    
    positionRef.current = newPosition;
    containerRef.current.style.transform = `translateX(${newPosition}px)`;
  };

  return (
    <div className="min-h-screen overflow-x-hidden bg-black">
      {/* Fixed Particle Background with CNC image, grid, vignette, and particles */}
      <ParticleBackground />
      
      {/* Navigation */}
      <Navigation />

      {/* Hero Section - Enhanced Gemini Style */}
      <section className="relative min-h-screen flex flex-col justify-center overflow-hidden -mt-16 pt-16">
        <div className="container-custom relative z-10 flex-1 flex items-center">
          <div className="max-w-4xl">
            {/* Red accent bar */}
            <AnimatedSection animation="fadeUp" delay={0}>
              <div className="flex items-center gap-3 mb-6">
                <div className="w-8 h-1 bg-primary rounded-full"></div>
                <span className="text-sm font-mono tracking-[0.2em] text-gray-400 uppercase">
                  Advanced Manufacturing Solutions
                </span>
              </div>
            </AnimatedSection>

            <AnimatedSection animation="fadeUp" delay={100}>
              <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold mb-2 leading-tight tracking-tight">
                <span className="bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                  Custom Manufacturing
                </span>
              </h1>
            </AnimatedSection>

            <AnimatedSection animation="fadeUp" delay={200}>
              <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-8 leading-tight tracking-tight">
                <span className="text-outline-white">
                  From Concept to Completion
                </span>
              </h2>
            </AnimatedSection>

            <AnimatedSection animation="fadeUp" delay={300}>
              <p className="text-lg md:text-xl text-gray-400 mb-10 max-w-lg leading-relaxed">
                Precision engineering and turnkey manufacturing solutions for complex parts and assemblies.
              </p>
            </AnimatedSection>

            <AnimatedSection animation="fadeUp" delay={400}>
              <div className="flex flex-col sm:flex-row gap-4">
                <Button size="lg" className="group uppercase tracking-wider" asChild>
                  <Link to="/contact">
                    Request a Quote
                    <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
                  </Link>
                </Button>
                <Button size="lg" variant="outline-light" className="uppercase tracking-wider" asChild>
                  <Link to="/capabilities">Explore Capabilities</Link>
                </Button>
              </div>
            </AnimatedSection>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-12 left-1/2 -translate-x-1/2 z-20">
          <ScrollIndicator />
        </div>
      </section>

      {/* Services Section - Transparent with glassmorphism cards */}
      <section className="relative z-10 py-24">
        <div className="container-custom">
          <AnimatedSection animation="fadeUp" className="mb-16">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-1 bg-primary rounded-full"></div>
              <span className="text-sm font-mono tracking-[0.2em] text-gray-400 uppercase">
                What We Do
              </span>
            </div>
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white">Our Services</h2>
          </AnimatedSection>

          <div className="grid md:grid-cols-2 gap-8">
            {services.map((service, index) => (
              <AnimatedSection key={index} animation="fadeUp" delay={index * 150}>
                <div className="bg-black/40 backdrop-blur-md border border-white/10 rounded-xl p-8 hover:border-primary/50 transition-all duration-500 group">
                  <h3 className="text-2xl md:text-3xl font-bold text-white mb-4 group-hover:text-primary transition-colors">
                    {service.title}
                  </h3>
                  <p className="text-gray-400 text-lg leading-relaxed mb-6">{service.description}</p>
                  <Button variant="outline-light" className="group/btn uppercase tracking-wider" asChild>
                    <Link to="/services">
                      Learn More
                      <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover/btn:translate-x-1" />
                    </Link>
                  </Button>
                  {/* Red bottom bar */}
                  <div className="mt-6 w-full h-0.5 bg-primary scale-x-0 group-hover:scale-x-100 transition-transform duration-500 origin-left" />
                </div>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* Capabilities Showcase - Horizontal Marquee */}
      <section className="relative z-10 py-24 border-t border-white/10 overflow-hidden">
        {/* Section Header */}
        <div className="container-custom mb-12">
          <AnimatedSection animation="fadeUp">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-1 bg-primary rounded-full"></div>
              <span className="text-sm font-mono tracking-[0.2em] text-gray-400 uppercase">
                Our Capabilities
              </span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-white">Precision Manufacturing</h2>
          </AnimatedSection>
        </div>

        {/* Infinite Scroll Strip */}
        <div
          ref={marqueeRef}
          className="relative w-full overflow-x-auto group cursor-grab select-none"
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onMouseMove={handleMouseMove}
          style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
        >
          <div
            ref={containerRef}
            className="flex whitespace-nowrap select-none"
            style={{ transform: 'translateX(0px)' }}
          >
            {/* First set of items */}
            {capabilities.map((capability, index) => (
              <div
                key={`first-${index}`}
                className="w-[350px] md:w-[450px] h-[380px] md:h-[420px] mx-3 md:mx-4 relative shrink-0 rounded-lg overflow-hidden border border-white/10 hover:border-primary/50 shadow-[0_0_20px_rgba(0,0,0,0.3)] hover:shadow-[0_0_30px_rgba(220,38,38,0.2)] transition-all duration-500 group/card cursor-pointer"
              >
                <div
                  className={`absolute inset-0 bg-no-repeat grayscale brightness-75 group-hover/card:grayscale-0 group-hover/card:brightness-100 transition-[filter,transform] duration-700 scale-100 group-hover/card:scale-110 ${capability.imageStyle || "bg-cover bg-center"}`}
                  style={{ backgroundImage: `url(${capability.image})` }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black via-black/60 to-transparent opacity-90 group-hover/card:opacity-80 transition-opacity duration-500" />
                
                {/* Capability badge */}
                <div className="absolute top-4 left-4 opacity-0 group-hover/card:opacity-100 transition-all duration-500 transform -translate-y-2 group-hover/card:translate-y-0">
                  <span className="text-xs font-mono tracking-wider text-primary bg-black/60 backdrop-blur-sm px-3 py-1 rounded-full border border-primary/30">
                    CAPABILITY
                  </span>
                </div>
                
                {/* Content - slides up on hover */}
                <div className="absolute bottom-0 left-0 right-0 p-6 transform transition-transform duration-500 group-hover/card:-translate-y-2">
                  <h3 className="text-xl md:text-2xl font-bold text-white mb-3 group-hover/card:text-primary transition-colors duration-300">{capability.title}</h3>
                  <p className="text-sm text-gray-400 whitespace-normal opacity-60 group-hover/card:opacity-100 transition-opacity duration-500 line-clamp-3">{capability.description}</p>
                </div>
                
                {/* Red bottom bar */}
                <div className="absolute bottom-0 left-0 w-full h-1 bg-primary scale-x-0 group-hover/card:scale-x-100 transition-transform duration-500 origin-left" />
              </div>
            ))}
            {/* Duplicate set for seamless loop */}
            {capabilities.map((capability, index) => (
              <div
                key={`second-${index}`}
                className="w-[350px] md:w-[450px] h-[380px] md:h-[420px] mx-3 md:mx-4 relative shrink-0 rounded-lg overflow-hidden border border-white/10 hover:border-primary/50 shadow-[0_0_20px_rgba(0,0,0,0.3)] hover:shadow-[0_0_30px_rgba(220,38,38,0.2)] transition-all duration-500 group/card cursor-pointer"
              >
                <div
                  className={`absolute inset-0 bg-no-repeat grayscale brightness-75 group-hover/card:grayscale-0 group-hover/card:brightness-100 transition-[filter,transform] duration-700 scale-100 group-hover/card:scale-110 ${capability.imageStyle || "bg-cover bg-center"}`}
                  style={{ backgroundImage: `url(${capability.image})` }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black via-black/60 to-transparent opacity-90 group-hover/card:opacity-80 transition-opacity duration-500" />
                
                {/* Capability badge */}
                <div className="absolute top-4 left-4 opacity-0 group-hover/card:opacity-100 transition-all duration-500 transform -translate-y-2 group-hover/card:translate-y-0">
                  <span className="text-xs font-mono tracking-wider text-primary bg-black/60 backdrop-blur-sm px-3 py-1 rounded-full border border-primary/30">
                    CAPABILITY
                  </span>
                </div>
                
                {/* Content - slides up on hover */}
                <div className="absolute bottom-0 left-0 right-0 p-6 transform transition-transform duration-500 group-hover/card:-translate-y-2">
                  <h3 className="text-xl md:text-2xl font-bold text-white mb-3 group-hover/card:text-primary transition-colors duration-300">{capability.title}</h3>
                  <p className="text-sm text-gray-400 whitespace-normal opacity-60 group-hover/card:opacity-100 transition-opacity duration-500 line-clamp-3">{capability.description}</p>
                </div>
                
                {/* Red bottom bar */}
                <div className="absolute bottom-0 left-0 w-full h-1 bg-primary scale-x-0 group-hover/card:scale-x-100 transition-transform duration-500 origin-left" />
              </div>
            ))}
          </div>
        </div>

        {/* View All Button */}
        <div className="container-custom mt-12 text-center">
          <AnimatedSection animation="fadeUp" delay={200}>
            <Button variant="outline-light" className="group uppercase tracking-wider" asChild>
              <Link to="/capabilities">
                Explore All Capabilities{" "}
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Link>
            </Button>
          </AnimatedSection>
        </div>
      </section>

      {/* Process Section - Gemini Design */}
      <section className="relative z-10 py-20 md:py-32 border-t border-white/10">
        <div className="container-custom">
          <AnimatedSection className="text-center max-w-3xl mx-auto mb-16">
            <div className="flex items-center justify-center gap-3 mb-4">
              <div className="w-8 h-1 bg-primary rounded-full"></div>
              <span className="text-sm font-mono tracking-[0.2em] text-gray-400 uppercase">
                How It Works
              </span>
              <div className="w-8 h-1 bg-primary rounded-full"></div>
            </div>
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-6">Our Process</h2>
            <p className="text-lg text-gray-400">
              From initial upload to final delivery, we streamline every step of your manufacturing journey.
            </p>
          </AnimatedSection>

          <ProcessSectionGemini className="max-w-6xl mx-auto" />
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 py-20 md:py-32 border-t border-white/10">
        <div className="container-custom text-center">
          <AnimatedSection animation="fadeUp">
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6">
              <span className="bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                Ready to Start?
              </span>
            </h2>
          </AnimatedSection>

          <AnimatedSection animation="fadeUp" delay={150}>
            <p className="text-xl text-gray-400 mb-10 max-w-lg mx-auto">
              Let's bring your manufacturing vision to life.
            </p>
          </AnimatedSection>

          <AnimatedSection animation="fadeUp" delay={300}>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" className="uppercase tracking-wider" asChild>
                <Link to="/contact">
                  Get a Quote <ArrowRight className="ml-2 h-5 w-5" />
                </Link>
              </Button>
              <Button size="lg" variant="outline-light" className="uppercase tracking-wider" asChild>
                <Link to="/projects">View Our Work</Link>
              </Button>
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative z-10 py-16 bg-black/60 backdrop-blur-md border-t border-white/10">
        <div className="container-custom">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8">
            {stats.map((stat, index) => (
              <AnimatedCounter key={index} value={stat.value} label={stat.label} delay={index * 100} />
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Index;
