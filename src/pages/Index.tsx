import { Link } from "react-router-dom";
import { ArrowRight, ChevronRight } from "lucide-react";
import { useRef, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import AnimatedSection from "@/components/home/AnimatedSection";
import AnimatedCounter from "@/components/home/AnimatedCounter";
import ParticleBackground from "@/components/home/ParticleBackground";
import Hero from "@/components/home/Hero";
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
      imageStyle: "bg-[length:80%] bg-center bg-no-repeat",
    },
    {
      title: "WIRE EDM",
      description: "Complex geometries with tight tolerances in hardened materials.",
      image: wireEdmImg,
      imageStyle: "bg-[length:75%] bg-center bg-no-repeat",
    },
    {
      title: "SHEET METAL",
      description: "Laser cutting, bending, and forming at any scale.",
      image: sheetMetalImg,
      imageStyle: "bg-[length:100%] bg-center bg-no-repeat",
    },
    {
      title: "HEAT TREATMENT",
      description: "Surface Hardening, PVD/CVD Coating, Chemical and Vacuum Heat Treatment",
      image: heatTreatmentImg,
      imageStyle: "bg-[length:80%] bg-center bg-no-repeat",
    },
    {
      title: "DIE CASTING",
      description: "Precision die casting service for customized metal parts",
      image: dieCastingImg,
    },
  ];

  // Marquee constants - EXACTLY as reference
  const CARD_WIDTH = 350;
  const GAP = 24;
  const ITEM_WIDTH = CARD_WIDTH + GAP;
  const SETS_TO_WRAP = 4;
  const WRAP_WIDTH = SETS_TO_WRAP * capabilities.length * ITEM_WIDTH;
  const AUTO_SPEED = 0.5;

  // Create 8x duplicated array for smooth infinite scroll
  const capabilitiesData = [
    ...capabilities,
    ...capabilities,
    ...capabilities,
    ...capabilities,
    ...capabilities,
    ...capabilities,
    ...capabilities,
    ...capabilities,
  ];

  // Drag to scroll state
  const marqueeRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [startX, setStartX] = useState(0);
  const dragStartPositionRef = useRef(0);

  // Animation refs
  const positionRef = useRef(0);
  const animationRef = useRef<number | null>(null);

  // Hover state with debouncing
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const hoverTimeoutRef = useRef<number | null>(null);

  const handleMouseEnter = (id: string) => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
    setHoveredId(id);
  };

  const handleMouseLeave = () => {
    hoverTimeoutRef.current = window.setTimeout(() => {
      setHoveredId(null);
    }, 100);
  };

  // Animation loop - EXACTLY as reference
  useEffect(() => {
    const animate = () => {
      if (!isDragging) {
        positionRef.current -= AUTO_SPEED;
      }

      // Real-time position wrapping
      if (positionRef.current <= -WRAP_WIDTH) {
        positionRef.current += WRAP_WIDTH;
      } else if (positionRef.current > 0) {
        positionRef.current -= WRAP_WIDTH;
      }

      if (containerRef.current) {
        containerRef.current.style.transform = `translate3d(${positionRef.current}px, 0, 0)`;
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    };
  }, [isDragging, WRAP_WIDTH]);

  const handleMouseDown = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDragging(true);
    const clientX = "touches" in e ? e.touches[0].clientX : (e as React.MouseEvent).clientX;
    setStartX(clientX);
    dragStartPositionRef.current = positionRef.current;
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleDragMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDragging) return;

    const clientX = "touches" in e ? e.touches[0].clientX : (e as React.MouseEvent).clientX;
    const delta = clientX - startX;
    positionRef.current = dragStartPositionRef.current + delta;

    // Real-time position wrapping during drag
    if (positionRef.current <= -WRAP_WIDTH) {
      positionRef.current += WRAP_WIDTH;
      dragStartPositionRef.current += WRAP_WIDTH;
    } else if (positionRef.current > 0) {
      positionRef.current -= WRAP_WIDTH;
      dragStartPositionRef.current -= WRAP_WIDTH;
    }

    if (containerRef.current) {
      containerRef.current.style.transform = `translate3d(${positionRef.current}px, 0, 0)`;
    }
  };

  return (
    <div className="min-h-screen overflow-x-hidden bg-black">
      {/* Fixed Particle Background with CNC image, grid, vignette, and particles */}
      <ParticleBackground />

      {/* Navigation */}
      <Navigation />

      {/* Hero Section */}
      <Hero />

      {/* Services Section */}
      <section className="relative z-10 py-24 bg-transparent border-t border-white/5">
        <div className="container-custom">
          <AnimatedSection animation="fadeUp" className="mb-16">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-1 bg-primary rounded-full"></div>
              <span className="text-sm font-mono tracking-[0.2em] text-gray-400 uppercase">What We Do</span>
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
      <section className="relative z-10 py-24 bg-transparent border-t border-white/5 overflow-hidden">
        {/* Section Header */}
        <div className="container-custom mb-12">
          <AnimatedSection animation="fadeUp">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-1 bg-primary rounded-full"></div>
              <span className="text-sm font-mono tracking-[0.2em] text-gray-400 uppercase">Our Capabilities</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-white">Precision Manufacturing</h2>
          </AnimatedSection>
        </div>

        {/* Infinite Scroll Strip */}
        <div
          ref={marqueeRef}
          className="relative w-full overflow-hidden cursor-grab select-none"
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onMouseMove={handleDragMove}
          onTouchStart={handleMouseDown}
          onTouchEnd={handleMouseUp}
          onTouchMove={handleDragMove}
        >
          <div
            ref={containerRef}
            className="flex w-max"
            style={{
              gap: `${GAP}px`,
              transform: "translate3d(0px, 0, 0)",
              backfaceVisibility: "hidden",
              willChange: "transform",
            }}
          >
            {capabilitiesData.map((capability, index) => {
              const isHovered = hoveredId === capability.title;
              return (
                <div
                  key={`${capability.title}-${index}`}
                  onMouseEnter={() => handleMouseEnter(capability.title)}
                  onMouseLeave={handleMouseLeave}
                  className={`relative flex-shrink-0 overflow-hidden rounded-sm border select-none backdrop-blur-sm transition-colors duration-300 cursor-pointer ${
                    isHovered ? "border-primary" : "border-white/10"
                  }`}
                  style={{
                    height: "400px",
                    width: `${CARD_WIDTH}px`,
                    backgroundColor: "rgba(5, 5, 5, 0.4)",
                  }}
                >
                  <div
                    className={`absolute inset-0 bg-cover bg-center transition-all duration-700 pointer-events-none ${
                      isHovered ? "grayscale-0 scale-110" : "grayscale"
                    } ${capability.imageStyle || ""}`}
                    style={{ backgroundImage: `url(${capability.image})` }}
                  />
                  <div
                    className={`absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent transition-opacity duration-300 pointer-events-none ${
                      isHovered ? "opacity-0" : "opacity-90"
                    }`}
                  />

                  <div
                    className={`absolute bottom-0 left-0 p-6 w-full transition-transform duration-300 pointer-events-none ${
                      isHovered ? "-translate-y-0" : "translate-y-2"
                    }`}
                  >
                    <span
                      className={`text-primary text-xs font-bold uppercase tracking-widest mb-2 block transition-opacity duration-300 ${
                        isHovered ? "opacity-100" : "opacity-0"
                      }`}
                    >
                      Capability
                    </span>
                    <h3 className="text-2xl font-bold text-white mb-2 uppercase">{capability.title}</h3>
                    <p
                      className={`text-gray-400 text-sm leading-relaxed whitespace-normal transition-opacity duration-300 ${
                        isHovered ? "opacity-100" : "opacity-0"
                      }`}
                    >
                      {capability.description}
                    </p>
                  </div>

                  <div
                    className={`absolute inset-0 border-2 border-primary transition-opacity duration-300 pointer-events-none rounded-sm ${
                      isHovered ? "opacity-100" : "opacity-0"
                    }`}
                  />
                </div>
              );
            })}
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

      {/* Process Section - Reference Design with Big Numbers */}
      <section className="relative z-10 py-24 bg-transparent border-t border-white/5">
        <div className="container mx-auto px-6">
          <div className="mb-16 max-w-2xl">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Streamlined Workflow</h2>
            <p className="text-gray-400 font-light">From digital file to physical part in record time.</p>
          </div>

          <div className="flex flex-col md:flex-row border-t border-white/10 bg-black/40 backdrop-blur-md rounded-sm">
            {[
              { id: "01", title: "Upload", text: "Submit your CAD files securely to our portal for instant parsing." },
              { id: "02", title: "Quote", text: "Get detailed pricing and lead time within 24 hours." },
              { id: "03", title: "Manufacture", text: "Production begins using high-speed 5-axis CNC centers." },
              { id: "04", title: "Quality", text: "Rigorous inspection ensures every part meets specifications." },
              { id: "05", title: "Deliver", text: "Parts are packed and shipped directly to your facility." },
            ].map((step, index) => (
              <AnimatedSection
                key={step.id}
                animation="fadeUp"
                delay={index * 100}
                className="flex-1 p-8 border-l border-white/10 first:border-l-0 border-b md:border-b-0 border-white/10 hover:bg-white/5 transition-colors group cursor-default relative"
              >
                <div className="absolute top-0 left-0 w-full h-[1px] bg-primary transform scale-x-0 group-hover:scale-x-100 transition-transform duration-500 origin-left" />

                <span className="block text-5xl font-bold text-white/10 group-hover:text-primary group-hover:translate-x-2 transition-all duration-300 mb-6">
                  {step.id}
                </span>
                <h3 className="text-xl font-bold text-white mb-3 uppercase tracking-wide">{step.title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed font-light">{step.text}</p>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section - Transparent Background */}
      <section className="relative z-10 py-20 bg-transparent">
        <div className="container-custom text-center">
          <AnimatedSection animation="fadeUp">
            <h2 className="text-3xl md:text-5xl font-black text-white uppercase mb-6 tracking-tight">
              Ready to Start?
            </h2>
          </AnimatedSection>

          <AnimatedSection animation="fadeUp" delay={150}>
            <p className="text-white/80 text-lg mb-8 max-w-2xl mx-auto">
              Let's bring your manufacturing vision to life. Get a quote in under 2 hours.
            </p>
          </AnimatedSection>

          <AnimatedSection animation="fadeUp" delay={300}>
            <Button
              size="lg"
              className="bg-primary text-white hover:bg-primary/90 uppercase tracking-widest shadow-xl"
              asChild
            >
              <Link to="/contact#send-message">Get a Quote</Link>
            </Button>
          </AnimatedSection>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Index;
