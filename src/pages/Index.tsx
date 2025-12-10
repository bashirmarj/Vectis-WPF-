import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import AnimatedSection from "@/components/home/AnimatedSection";
import AnimatedCounter from "@/components/home/AnimatedCounter";
import ParallaxSection from "@/components/home/ParallaxSection";
import ProcessTimeline from "@/components/home/ProcessTimeline";
import ScrollIndicator from "@/components/home/ScrollIndicator";
import { useStaggerAnimation } from "@/hooks/useScrollAnimation";
import heroImage from "@/assets/hero-cnc-turbine.png";
import customPartsImg from "@/assets/custom-parts-cnc.png";
import prototypeImg from "@/assets/prototype-cnc-part.png";
import capabilitiesImg from "@/assets/capabilities-bg.jpg";

const Index = () => {
  const services = [
    {
      title: "PROTOTYPE DESIGN",
      description: "From concept to functional prototype. Fast turnaround, precision engineering.",
      image: prototypeImg,
      link: "/services/prototype-design",
    },
    {
      title: "CUSTOM PARTS",
      description: "High-quality components manufactured to exact specifications.",
      image: customPartsImg,
      link: "/services/custom-parts",
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
      image: capabilitiesImg
    },
    { 
      title: "WIRE EDM", 
      description: "Complex geometries with tight tolerances in hardened materials.",
      image: capabilitiesImg
    },
    { 
      title: "SHEET METAL", 
      description: "Laser cutting, bending, and forming at any scale.",
      image: capabilitiesImg
    },
  ];

  const whyChooseUs = [
    {
      number: "01",
      title: "End-to-End Solutions",
      description: "From initial concept and design through manufacturing and final delivery, we manage your entire project lifecycle.",
    },
    {
      number: "02",
      title: "Precision Engineering",
      description: "State-of-the-art equipment and rigorous quality control ensure every part meets exact specifications.",
    },
    {
      number: "03",
      title: "Expert Team",
      description: "Our experienced engineers and technicians bring deep industry knowledge to every project.",
    },
  ];

  const { ref: whyRef, isVisible: whyVisible } = useStaggerAnimation(whyChooseUs.length, 200);

  return (
    <div className="min-h-screen overflow-x-hidden bg-black">
      <Navigation />

      {/* Hero Section - Full Viewport */}
      <section className="relative min-h-[150vh] flex flex-col justify-center overflow-hidden -mt-16 pt-16">
        <ParallaxSection
          backgroundImage={heroImage}
          overlayOpacity={0.5}
          className="absolute inset-0"
          contentClassName="hidden"
        >
          <div />
        </ParallaxSection>
        
        <div className="container-custom relative z-10 flex-1 flex items-center">
          <div className="max-w-3xl">
            <AnimatedSection animation="fadeUp" delay={0}>
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-semibold text-white mb-6 leading-tight tracking-tight">
                Custom Manufacturing
                <br />
                <span className="text-primary font-light">From Concept to Completion</span>
              </h1>
            </AnimatedSection>
            
            <AnimatedSection animation="fadeUp" delay={200}>
              <p className="text-lg md:text-xl text-gray-300 mb-10 max-w-lg leading-relaxed">
                Precision engineering and turnkey manufacturing solutions.
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

      {/* Services - Full-Bleed Vertical Sections */}
      {services.map((service, index) => {
        const isEven = index % 2 === 0;
        
        return (
          <section key={index} className="relative min-h-[150vh] flex items-center">
            {/* Background Image */}
            <ParallaxSection
              backgroundImage={service.image}
              overlayOpacity={0.4}
              className="absolute inset-0"
              contentClassName="hidden"
            >
              <div />
            </ParallaxSection>

            {/* Content */}
            <div className="container-custom relative z-10">
              <div className={`max-w-md ${isEven ? '' : 'ml-auto text-right'}`}>
                <AnimatedSection animation="fadeUp" delay={0}>
                  <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6 uppercase tracking-wide">
                    {service.title}
                  </h2>
                </AnimatedSection>
                
                <AnimatedSection animation="fadeUp" delay={150}>
                  <p className="text-lg text-gray-300 mb-8 leading-relaxed">
                    {service.description}
                  </p>
                </AnimatedSection>
                
                <AnimatedSection animation="fadeUp" delay={300}>
                  <Button size="lg" variant="outline-light" className="group uppercase tracking-wider" asChild>
                    <Link to={service.link}>
                      Learn More 
                      <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
                    </Link>
                  </Button>
                </AnimatedSection>
              </div>
            </div>
          </section>
        );
      })}

      {/* Capabilities Showcase - Horizontal Marquee with Grayscale-to-Color Effect */}
      <section className="py-24 bg-neutral-950 border-y border-white/10 overflow-hidden">
        {/* Section Header */}
        <div className="container-custom mb-12">
          <AnimatedSection animation="fadeUp">
            <p className="text-sm font-mono tracking-[0.3em] text-gray-500 uppercase mb-3">
              Our Capabilities
            </p>
            <h2 className="text-3xl md:text-4xl font-bold text-white">
              Precision Manufacturing
            </h2>
          </AnimatedSection>
        </div>
        
        {/* Infinite Scroll Strip */}
        <div className="relative w-full overflow-hidden group">
          <div className="flex whitespace-nowrap animate-marquee hover:[animation-play-state:paused]">
            {/* First set of items */}
            {capabilities.map((capability, index) => (
              <div 
                key={`first-${index}`}
                className="w-[350px] md:w-[450px] h-[280px] md:h-[320px] mx-3 md:mx-4 relative shrink-0 rounded-lg overflow-hidden border border-white/10 group/card cursor-pointer"
              >
                <div 
                  className="absolute inset-0 bg-cover bg-center grayscale group-hover/card:grayscale-0 transition-all duration-500 scale-100 group-hover/card:scale-105"
                  style={{ backgroundImage: `url(${capability.image})` }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
                <div className="absolute bottom-0 left-0 right-0 p-5">
                  <span className="text-xs font-mono tracking-wider text-primary/80 uppercase mb-2 block">
                    Capability
                  </span>
                  <h3 className="text-lg md:text-xl font-bold text-white mb-2">
                    {capability.title}
                  </h3>
                  <p className="text-sm text-gray-400 whitespace-normal line-clamp-2">
                    {capability.description}
                  </p>
                </div>
              </div>
            ))}
            {/* Duplicate set for seamless loop */}
            {capabilities.map((capability, index) => (
              <div 
                key={`second-${index}`}
                className="w-[350px] md:w-[450px] h-[280px] md:h-[320px] mx-3 md:mx-4 relative shrink-0 rounded-lg overflow-hidden border border-white/10 group/card cursor-pointer"
              >
                <div 
                  className="absolute inset-0 bg-cover bg-center grayscale group-hover/card:grayscale-0 transition-all duration-500 scale-100 group-hover/card:scale-105"
                  style={{ backgroundImage: `url(${capability.image})` }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
                <div className="absolute bottom-0 left-0 right-0 p-5">
                  <span className="text-xs font-mono tracking-wider text-primary/80 uppercase mb-2 block">
                    Capability
                  </span>
                  <h3 className="text-lg md:text-xl font-bold text-white mb-2">
                    {capability.title}
                  </h3>
                  <p className="text-sm text-gray-400 whitespace-normal line-clamp-2">
                    {capability.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* View All Button */}
        <div className="container-custom mt-12 text-center">
          <AnimatedSection animation="fadeUp" delay={200}>
            <Button variant="outline-light" className="group uppercase tracking-wider" asChild>
              <Link to="/capabilities">
                Explore All Capabilities <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Link>
            </Button>
          </AnimatedSection>
        </div>
      </section>

      {/* Process Timeline */}
      <section className="py-20 md:py-32 bg-background">
        <div className="container-custom">
          <AnimatedSection className="text-center max-w-3xl mx-auto mb-16">
            <p className="text-primary font-semibold text-sm uppercase tracking-[0.2em] mb-4">
              How It Works
            </p>
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-6">
              Our Process
            </h2>
            <p className="text-lg text-muted-foreground">
              From initial upload to final delivery, we streamline every step 
              of your manufacturing journey.
            </p>
          </AnimatedSection>

          <ProcessTimeline className="max-w-5xl mx-auto" />
        </div>
      </section>

      {/* Why Choose Vectis */}
      <section className="py-20 md:py-32 bg-background">
        <div className="container-custom">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-20 items-center">
            <div>
              <AnimatedSection animation="fadeLeft">
                <p className="text-primary font-semibold text-sm uppercase tracking-[0.2em] mb-4">
                  Why Vectis
                </p>
                <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-8">
                  Built on Precision,
                  <br />
                  Driven by Excellence
                </h2>
              </AnimatedSection>

              <div ref={whyRef} className="space-y-8">
                {whyChooseUs.map((item, index) => (
                  <div
                    key={index}
                    className={`flex gap-5 transition-all duration-700 ${
                      whyVisible
                        ? 'opacity-100 translate-x-0'
                        : 'opacity-0 -translate-x-8'
                    }`}
                    style={{ transitionDelay: `${index * 200}ms` }}
                  >
                    <div className="flex-shrink-0 w-14 h-14 bg-primary rounded-lg flex items-center justify-center">
                      <span className="text-primary-foreground font-bold text-lg">
                        {item.number}
                      </span>
                    </div>
                    <div>
                      <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                      <p className="text-muted-foreground leading-relaxed">
                        {item.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <AnimatedSection animation="fadeUp" delay={600} className="mt-10">
                <Button size="lg" asChild>
                  <Link to="/about">Learn About Us</Link>
                </Button>
              </AnimatedSection>
            </div>

            <AnimatedSection animation="fadeRight" className="relative">
              <div className="relative rounded-lg overflow-hidden shadow-2xl">
                <img
                  src={capabilitiesImg}
                  alt="Manufacturing facility"
                  className="w-full h-auto"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-accent/40 to-transparent" />
              </div>
              {/* Floating accent element */}
              <div className="absolute -bottom-6 -left-6 w-32 h-32 bg-primary/10 rounded-lg -z-10" />
              <div className="absolute -top-6 -right-6 w-24 h-24 bg-primary/5 rounded-lg -z-10" />
            </AnimatedSection>
          </div>
        </div>
      </section>

      {/* CTA Section with Parallax */}
      <section className="relative min-h-[150vh] flex items-center">
        <ParallaxSection
          backgroundImage={capabilitiesImg}
          overlayOpacity={0.5}
          className="absolute inset-0"
          contentClassName="hidden"
        >
          <div />
        </ParallaxSection>
        
        <div className="container-custom relative z-10 text-center">
          <AnimatedSection animation="fadeUp">
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6 uppercase tracking-wide">
              Ready to Start?
            </h2>
          </AnimatedSection>
          
          <AnimatedSection animation="fadeUp" delay={150}>
            <p className="text-xl text-gray-300 mb-10 max-w-lg mx-auto">
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

      {/* Stats Section - At bottom before footer */}
      <section className="py-16 bg-accent/50 border-t border-border/30">
        <div className="container-custom">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8">
            {stats.map((stat, index) => (
              <AnimatedCounter
                key={index}
                value={stat.value}
                label={stat.label}
                delay={index * 100}
              />
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Index;
