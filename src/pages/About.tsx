import { Target, Award, Users, TrendingUp } from "lucide-react";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Card, CardContent } from "@/components/ui/card";
import AnimatedSection from "@/components/home/AnimatedSection";
import ParticleBackground from "@/components/home/ParticleBackground";
import aboutManufacturingImg from "@/assets/about-manufacturing.jpg";
import capabilitiesImg from "@/assets/capabilities-bg.jpg";
const About = () => {
  const values = [{
    icon: Target,
    title: "Precision First",
    description: "Every component manufactured to exacting standards with rigorous quality control."
  }, {
    icon: Award,
    title: "Engineering Excellence",
    description: "Decades of expertise combined with cutting-edge manufacturing technology."
  }, {
    icon: Users,
    title: "Customer Partnership",
    description: "Collaborative approach ensuring your vision becomes reality on time and on budget."
  }, {
    icon: TrendingUp,
    title: "Continuous Innovation",
    description: "Investing in the latest technology and techniques to deliver superior results."
  }];
  return <div className="min-h-screen bg-black">
      <Navigation />
      <ParticleBackground />

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden z-10">
        <div className="absolute inset-0 opacity-20 bg-cover bg-center" style={{
        backgroundImage: `url(${capabilitiesImg})`
      }}></div>
        <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-black/40 to-black"></div>
        <div className="container-custom relative z-10">
          <AnimatedSection animation="fadeUp">
            <div className="max-w-3xl">
              <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 leading-snug tracking-normal">
                About Vectis Manufacturing
              </h1>
              <p className="text-xl text-gray-300 leading-relaxed">
                For over 25 years, Vectis has been at the forefront of precision manufacturing, delivering innovative solutions that drive our clients' success.
              </p>
              
              {/* Brand Meaning Callout */}
              <div className="mt-8 border-l-4 border-primary bg-black/40 backdrop-blur-md rounded-r-lg p-6 max-w-2xl">
                <h3 className="text-sm font-semibold text-primary mb-3 tracking-wide uppercase font-mono">The Vectis Name</h3>
                <p className="text-gray-300 italic leading-relaxed">
                  Derived from the Latin word for lever, Vectis embodies precision, strength, and intelligent design. Like a lever that multiplies force, Vectis Manufacturing transforms engineering expertise into powerful, efficient solutions that drive performance and reliability.
                </p>
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* Mission & Vision */}
      <section className="relative z-10 border-t border-white/5">
        <div className="container-custom section-spacing">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <AnimatedSection animation="fadeRight">
              <img src={aboutManufacturingImg} alt="Precision manufacturing equipment at Vectis facility" className="rounded-lg shadow-xl border border-white/10" />
            </AnimatedSection>
            <AnimatedSection animation="fadeLeft" delay={100}>
              <div className="space-y-8">
                <div>
                  <div className="flex items-start gap-2 mb-4">
                    <div className="w-1 h-8 bg-primary rounded-full"></div>
                    <h2 className="text-3xl font-bold text-white tracking-tight font-mono uppercase">Our Mission</h2>
                  </div>
                  <p className="text-lg text-gray-300 leading-relaxed">
                    To deliver exceptional manufacturing solutions that exceed our clients' expectations through precision engineering, innovative technology, and unwavering commitment to quality. We transform concepts into reality, supporting businesses across industries with reliable, scalable manufacturing partnerships.
                  </p>
                </div>
                <div>
                  <div className="flex items-start gap-2 mb-4">
                    <div className="w-1 h-8 bg-primary rounded-full"></div>
                    <h2 className="text-3xl font-bold text-white tracking-tight font-mono uppercase">Our Vision</h2>
                  </div>
                  <p className="text-lg text-gray-300 leading-relaxed">
                    To be the preferred manufacturing partner for companies seeking precision, reliability, and innovation. We envision a future where advanced manufacturing technology and expert craftsmanship combine to solve complex engineering challenges and drive industrial progress.
                  </p>
                </div>
              </div>
            </AnimatedSection>
          </div>
        </div>
      </section>

      {/* Core Values */}
      <section className="relative z-10 border-t border-white/5">
        <div className="container-custom section-spacing">
          <AnimatedSection animation="fadeUp">
            <div className="text-center max-w-3xl mx-auto mb-16">
              <div className="flex items-start gap-2 mb-4 justify-center">
                <div className="w-1 h-8 bg-primary rounded-full"></div>
                <h2 className="text-3xl font-bold text-white tracking-tight font-mono uppercase">Our Core Values</h2>
              </div>
              <p className="text-lg text-gray-400">
                The principles that guide every project and partnership we undertake.
              </p>
            </div>
          </AnimatedSection>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {values.map((value, index) => {
            const Icon = value.icon;
            return <AnimatedSection key={index} animation="fadeUp" delay={index * 100}>
                  <Card className="h-full bg-black/40 backdrop-blur-md border border-white/10 hover:border-primary/50 transition-all duration-300 hover:-translate-y-1">
                    <CardContent className="p-6">
                      <div className="w-14 h-14 bg-primary/20 rounded-lg flex items-center justify-center mb-4">
                        <Icon className="h-7 w-7 text-primary" />
                      </div>
                      <h3 className="text-xl font-bold text-white mb-2">{value.title}</h3>
                      <p className="text-gray-400">{value.description}</p>
                    </CardContent>
                  </Card>
                </AnimatedSection>;
          })}
          </div>
        </div>
      </section>

      {/* Company Stats */}
      <section className="relative z-10 border-t border-white/5">
        
      </section>

      {/* Expertise Section */}
      <section className="relative z-10 border-t border-white/5">
        <div className="container-custom section-spacing">
          <AnimatedSection animation="fadeUp">
            <div className="max-w-3xl mx-auto text-center">
              <div className="flex items-start gap-2 mb-4 justify-center">
                <div className="w-1 h-8 bg-primary rounded-full"></div>
                <h2 className="text-3xl font-bold text-white tracking-tight font-mono uppercase">Expertise Across Industries</h2>
              </div>
              <p className="text-lg text-gray-400 mb-12">
                Our diverse experience spans multiple sectors, giving us unique insights into the specific challenges and requirements of each industry we serve.
              </p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {["Aerospace", "Automotive", "Medical Devices", "Industrial Equipment", "Energy & Power", "Defense & Military"].map((industry, index) => <AnimatedSection key={industry} animation="scaleIn" delay={index * 50}>
                    <div className="bg-black/40 backdrop-blur-md border border-white/10 rounded-lg p-4 font-semibold text-center text-gray-300 hover:border-primary/50 hover:text-primary transition-colors">
                      {industry}
                    </div>
                  </AnimatedSection>)}
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>

      <Footer />
    </div>;
};
export default About;