import { Link } from "react-router-dom";
import { ArrowRight, ArrowUpRight, X, Check } from "lucide-react";
import { useRef, useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import AnimatedSection from "@/components/home/AnimatedSection";
import AnimatedCounter from "@/components/home/AnimatedCounter";
import ParticleBackground from "@/components/home/ParticleBackground";
import Hero from "@/components/home/Hero";
import WorkflowSection from "@/components/home/WorkflowSection";
import cncMachiningImg from "@/assets/cnc-machining-black.png";
import cncMachiningDetailImg from "@/assets/cnc-machining-detail.png";
import sheetMetalImg from "@/assets/sheet-metal-black.png";
import sheetMetalDetailImg from "@/assets/sheet-metal-detail.png";
import heatTreatmentImg from "@/assets/heat-treatment-black.png";
import dieCastingImg from "@/assets/die-casting-black.png";
import wireEdmImg from "@/assets/wire-edm-black.png";

interface Capability {
  id: string;
  title: string;
  shortDescription: string;
  fullDescription: string;
  features: string[];
  image: string;
  detailImage: string;
  imageStyle?: string;
}

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

  const capabilities: Capability[] = [
    {
      id: "cnc-machining",
      title: "CNC MACHINING",
      shortDescription: "Precision 3, 4, & 5-axis milling for complex metal and plastic parts.",
      fullDescription:
        "Leverage our advanced multi-axis machining centers for unparalleled precision. We offer automated turning and milling services optimized for both rapid prototyping and full-scale production runs. Our automated lines ensure consistency across thousands of parts.",
      features: [
        "3, 4, & 5-axis simultaneous milling",
        "Tolerances down to ±0.01mm",
        "Instant DFM & Quotes",
        "Production as fast as 1 day",
        "Metals: Aluminum, Steel, Titanium, Brass",
        "Plastics: ABS, PEEK, POM, Nylon",
      ],
      image: cncMachiningImg,
      detailImage: cncMachiningDetailImg,
      imageStyle: "bg-[length:80%] bg-center bg-no-repeat",
    },
    {
      id: "sheet-metal",
      title: "SHEET METAL",
      shortDescription: "Laser cutting, bending, and assembly for durable enclosures.",
      fullDescription:
        "Our sheet metal fabrication services cover the entire production chain. We utilize high-power fiber lasers for clean cuts, CNC brakes for precise bending, and offer full assembly services including welding, riveting, and PEM insertion.",
      features: [
        "Laser Cutting & CNC Punching",
        "Precision Bending & Forming",
        "TIG, MIG, & Spot Welding",
        "Hardware Insertion & Riveting",
        "Powder Coating & Anodizing",
        "Materials: Aluminum, Steel, Copper",
      ],
      image: sheetMetalImg,
      detailImage: sheetMetalDetailImg,
      imageStyle: "bg-[length:81%] bg-center bg-no-repeat",
    },
    {
      id: "wire-edm",
      title: "WIRE EDM",
      shortDescription: "Complex geometries with tight tolerances in hardened materials.",
      fullDescription:
        "Our Wire EDM service delivers exceptional precision for intricate cuts in hardened materials. Ideal for aerospace, medical, and tooling applications where conventional machining cannot achieve the required tolerances or geometries.",
      features: [
        "Tolerances to ±0.002mm",
        "Any conductive material",
        "Complex internal features",
        "No mechanical stress on workpiece",
        "Hardened tool steels",
        "Carbide & exotic alloys",
      ],
      image: wireEdmImg,
      detailImage: "https://images.unsplash.com/photo-1565043589221-1a6fd9ae45c7?q=80&w=1000&auto=format&fit=crop",
      imageStyle: "bg-[length:65%] bg-center bg-no-repeat",
    },
    {
      id: "heat-treatment",
      title: "HEAT TREATMENT",
      shortDescription: "Surface Hardening, PVD/CVD Coating, Chemical and Vacuum Heat Treatment",
      fullDescription:
        "Comprehensive heat treatment services to enhance material properties. From surface hardening to vacuum treatment, we optimize parts for durability, wear resistance, and performance in demanding applications.",
      features: [
        "Surface Hardening",
        "PVD/CVD Coating",
        "Vacuum Heat Treatment",
        "Chemical Treatment",
        "Stress Relief Annealing",
        "Hardness Testing & QC",
      ],
      image: heatTreatmentImg,
      detailImage: "https://images.unsplash.com/photo-1588622180862-23b006730595?q=80&w=1000&auto=format&fit=crop",
      imageStyle: "bg-[length:80%] bg-center bg-no-repeat",
    },
    {
      id: "die-casting",
      title: "DIE CASTING",
      shortDescription: "High-volume metal parts with excellent surface finish.",
      fullDescription:
        "Ideal for high-volume production of metal components. Our high-pressure die casting process produces parts with near-net shape, reducing the need for post-machining. Perfect for automotive and consumer electronics housings.",
      features: [
        "Materials: Aluminum A380, ADC12, Zinc",
        "High-pressure die casting",
        "Hot & Cold chamber machines",
        "Excellent surface consistency",
        "Post-machining & finishing included",
        "Scalable to 100,000+ units",
      ],
      image: dieCastingImg,
      detailImage: "https://images.unsplash.com/photo-1533237264842-83675a61fb8c?q=80&w=1000&auto=format&fit=crop",
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
  const hasMovedRef = useRef(false);

  // Animation refs
  const positionRef = useRef(0);
  const animationRef = useRef<number | null>(null);

  // Hover state with debouncing
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const hoverTimeoutRef = useRef<number | null>(null);

  // Modal state
  const [selectedCapability, setSelectedCapability] = useState<Capability | null>(null);

  const handleCardClick = (capability: Capability) => {
    if (!hasMovedRef.current) {
      setSelectedCapability(capability);
    }
  };

  const closeModal = () => setSelectedCapability(null);

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

  // Animation loop - pause when modal is open
  useEffect(() => {
    const animate = () => {
      if (!isDragging && !selectedCapability) {
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
  }, [isDragging, WRAP_WIDTH, selectedCapability]);

  const handleMouseDown = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDragging(true);
    hasMovedRef.current = false;
    const clientX = "touches" in e ? e.touches[0].clientX : (e as React.MouseEvent).clientX;
    setStartX(clientX);
    dragStartPositionRef.current = positionRef.current;
  };

  const handleMouseUp = () => {
    setTimeout(() => setIsDragging(false), 50);
  };

  const handleDragMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDragging) return;

    const clientX = "touches" in e ? e.touches[0].clientX : (e as React.MouseEvent).clientX;
    const delta = clientX - startX;

    if (Math.abs(delta) > 5) {
      hasMovedRef.current = true;
    }

    positionRef.current = dragStartPositionRef.current + delta;

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
    <div className="min-h-screen overflow-x-hidden bg-[#F5F5F2]">
      {/* Fixed Particle Background with CNC image, grid, vignette, and particles */}
      <ParticleBackground />

      {/* Navigation */}
      <Navigation />

      {/* Hero Section */}
      <Hero />

      {/* Services Section */}
      <section className="relative z-10 py-24 bg-transparent border-t border-slate-200">
        <div className="container-custom">
          <AnimatedSection animation="fadeUp" className="mb-16">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-1 bg-primary rounded-full"></div>
              <span className="text-sm font-mono tracking-[0.2em] text-slate-600 uppercase">What We Do</span>
            </div>
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-slate-900">Our Services</h2>
          </AnimatedSection>

          <div className="grid md:grid-cols-2 gap-8">
            {services.map((service, index) => (
              <AnimatedSection key={index} animation="fadeUp" delay={index * 150}>
                <div className="h-full bg-[#F5F5F2] shadow-md border border-slate-200 rounded-xl p-8 hover:border-primary/50 transition-all duration-500 group flex flex-col">
                  <h3 className="text-2xl md:text-3xl font-bold text-slate-900 mb-4 group-hover:text-primary transition-colors">
                    {service.title}
                  </h3>
                  <p className="text-slate-600 text-lg leading-relaxed flex-grow">{service.description}</p>
                  {/* Red bottom bar */}
                  <div className="mt-6 w-full h-0.5 bg-primary scale-x-0 group-hover:scale-x-100 transition-transform duration-500 origin-left" />
                </div>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* Capabilities Showcase - Horizontal Marquee */}
      <section id="capabilities" className="relative z-10 py-24 bg-transparent border-t border-slate-200 overflow-hidden">
        {/* Section Header */}
        <div className="container-custom mb-12">
          <AnimatedSection animation="fadeUp">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-1 bg-primary rounded-full"></div>
              <span className="text-sm font-mono tracking-[0.2em] text-slate-900 uppercase">Our Capabilities</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-slate-900">Precision Manufacturing</h2>
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
              const isHovered = hoveredId === capability.id;
              return (
                <div
                  key={`${capability.id}-${index}`}
                  onMouseEnter={() => handleMouseEnter(capability.id)}
                  onMouseLeave={handleMouseLeave}
                  onClick={() => handleCardClick(capability)}
                  className="relative flex-shrink-0 overflow-hidden rounded-xl border border-primary/40 select-none shadow-[0_0_15px_rgba(200,30,50,0.15)] transition-all duration-300 cursor-pointer flex flex-col"
                  style={{
                    height: "400px",
                    width: `${CARD_WIDTH}px`,
                    backgroundColor: "hsl(var(--card))",
                  }}
                >
                  {/* Image Section */}
                  <div className="relative h-56 overflow-hidden bg-black">
                    <div
                      className={`absolute inset-0 bg-center bg-no-repeat transition-all duration-700 pointer-events-none ${
                        isHovered ? "grayscale-0 scale-110" : "grayscale"
                      }`}
                      style={{
                        backgroundImage: `url(${capability.image}?v=2.0)`,
                        backgroundSize:
                          capability.id === "cnc-machining"
                            ? "70%"
                            : capability.id === "wire-edm"
                              ? "55%"
                              : capability.id === "sheet-metal"
                                ? "90%"
                                : capability.id === "heat-treatment"
                                  ? "80%"
                                  : "cover",
                      }}
                    />
                  </div>

                  {/* Text Section */}
                  <div className="flex-1 p-6 flex flex-col">
                    <div className="flex justify-between items-center mb-3">
                      <span className="text-primary text-xs font-bold uppercase tracking-widest">Capability</span>
                      <ArrowUpRight
                        className={`w-4 h-4 text-primary transition-transform duration-300 ${isHovered ? "translate-x-1 -translate-y-1" : ""}`}
                      />
                    </div>
                    <h3
                      className={`text-xl font-bold mb-2 uppercase transition-colors duration-300 ${isHovered ? "text-primary" : "text-foreground"}`}
                    >
                      {capability.title}
                    </h3>
                    <p className="text-muted-foreground text-sm leading-relaxed whitespace-normal">
                      {capability.shortDescription}
                    </p>
                  </div>

                  {/* Bottom accent line */}
                  <div
                    className={`absolute bottom-0 left-0 right-0 h-0.5 bg-primary transform transition-transform duration-500 origin-left ${
                      isHovered ? "scale-x-100" : "scale-x-0"
                    }`}
                  />
                </div>
              );
            })}
          </div>
        </div>

      </section>

      {/* Process Section - Reference Design with Big Numbers */}
      <WorkflowSection />

      {/* CTA Section - Transparent Background */}
      <section className="relative z-10 py-20 bg-transparent">
        <div className="container-custom text-center">
          <AnimatedSection animation="fadeUp">
            <h2 className="text-3xl md:text-5xl font-black text-slate-900 uppercase mb-6 tracking-tight">
              Ready to Start?
            </h2>
          </AnimatedSection>

          <AnimatedSection animation="fadeUp" delay={150}>
            <p className="text-slate-900 text-lg mb-8 max-w-2xl mx-auto">
              Let's bring your manufacturing vision to life. Get a quote in under 2 hours.
            </p>
          </AnimatedSection>

          <AnimatedSection animation="fadeUp" delay={300}>
            <Button
              size="lg"
              className="bg-primary text-white hover:bg-primary/90 uppercase tracking-widest shadow-lg"
              asChild
            >
              <Link to="/contact#send-message">Get a Quote</Link>
            </Button>
          </AnimatedSection>
        </div>
      </section>

      <Footer />

      {/* Capability Detail Modal */}
      <AnimatePresence>
        {selectedCapability && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 md:p-8">
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={closeModal}
              className="absolute inset-0 bg-slate-900/80 backdrop-blur-md"
            />

            {/* Modal Content */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-5xl bg-[#F5F5F2] border border-slate-200 rounded-sm shadow-2xl flex flex-col md:flex-row max-h-[75vh] overflow-hidden"
            >
              {/* Close Button */}
              <button
                onClick={closeModal}
                className="absolute top-4 right-4 z-20 p-2 bg-slate-100 text-slate-900 hover:bg-primary hover:text-white rounded-full transition-colors"
              >
                <X size={24} />
              </button>

              {/* Left Side: Image */}
              <div className="w-full md:w-2/5 h-52 md:h-auto relative shrink-0">
                <div
                  className="absolute inset-0 bg-cover bg-center"
                  style={{ backgroundImage: `url(${selectedCapability.detailImage})` }}
                />
                <div className="absolute inset-0 bg-gradient-to-t md:bg-gradient-to-r from-slate-900/60 to-transparent" />
                <div className="absolute bottom-6 left-6 text-white md:hidden">
                  <h3 className="text-3xl font-bold uppercase">{selectedCapability.title}</h3>
                </div>
              </div>

              {/* Right Side: Content */}
              <div className="w-full md:w-3/5 p-6 md:p-12 overflow-y-auto bg-[#F5F5F2]">
                <div className="hidden md:block mb-6">
                  <h3 className="text-4xl font-black text-slate-900 uppercase mb-2">{selectedCapability.title}</h3>
                  <div className="h-1 w-20 bg-primary" />
                </div>

                <p className="text-slate-600 text-lg leading-relaxed mb-8 font-light">
                  {selectedCapability.fullDescription}
                </p>

                <div className="mb-10">
                  <h4 className="text-slate-900 font-bold uppercase tracking-widest text-sm mb-4">
                    Key Features & Specs
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {selectedCapability.features.map((feature, idx) => (
                      <div key={idx} className="flex items-start gap-3">
                        <Check className="w-5 h-5 text-primary shrink-0 mt-0.5" />
                        <span className="text-slate-600 text-sm">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex flex-col sm:flex-row gap-4 pt-6 border-t border-slate-200">
                  <Button
                    className="flex-1 bg-primary hover:bg-primary/90 text-white py-4 px-6 uppercase tracking-wide"
                    asChild
                  >
                    <Link to="/contact#send-message">
                      Get Instant Quote <ArrowRight className="ml-2" size={18} />
                    </Link>
                  </Button>
                  <Button
                    onClick={closeModal}
                    variant="outline"
                    className="flex-1 bg-slate-100 hover:bg-slate-200 text-slate-900 py-4 px-6 uppercase tracking-wide border-slate-200"
                  >
                    Close Details
                  </Button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Index;
