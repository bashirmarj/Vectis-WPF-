import { Target, Award, Users, TrendingUp } from "lucide-react";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Card, CardContent } from "@/components/ui/card";
import AnimatedSection from "@/components/home/AnimatedSection";
import aboutManufacturingImg from "@/assets/about-manufacturing.jpg";
import capabilitiesImg from "@/assets/capabilities-bg.jpg";

const About = () => {
  const values = [
    {
      icon: Target,
      title: "Precision First",
      description: "Every component manufactured to exacting standards with rigorous quality control.",
    },
    {
      icon: Award,
      title: "Engineering Excellence",
      description: "Decades of expertise combined with cutting-edge manufacturing technology.",
    },
    {
      icon: Users,
      title: "Customer Partnership",
      description: "Collaborative approach ensuring your vision becomes reality on time and on budget.",
    },
    {
      icon: TrendingUp,
      title: "Continuous Innovation",
      description: "Investing in the latest technology and techniques to deliver superior results.",
    },
  ];

  return (
    <div className="min-h-screen">
      <Navigation />

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 bg-accent text-accent-foreground overflow-hidden">
        <div
          className="absolute inset-0 opacity-15 bg-cover bg-center"
          style={{ backgroundImage: `url(${capabilitiesImg})` }}
        ></div>
        <div className="container-custom relative z-10">
          <AnimatedSection animation="fadeUp">
            <div className="max-w-3xl">
              <h1 className="text-white mb-6">About Vectis Manufacturing</h1>
              <p className="text-xl text-gray-200 leading-relaxed">
                For over 25 years, Vectis has been at the forefront of precision manufacturing, delivering innovative solutions that drive our clients' success.
              </p>
              
              {/* Brand Meaning Callout */}
              <div className="mt-8 border-l-4 border-primary bg-white/5 rounded-r-lg p-6 max-w-2xl">
                <h3 className="text-sm font-semibold text-primary mb-3 tracking-wide uppercase">The Vectis Name</h3>
                <p className="text-gray-200 italic leading-relaxed">
                  Derived from the Latin word for lever, Vectis embodies precision, strength, and intelligent design. Like a lever that multiplies force, Vectis Manufacturing transforms engineering expertise into powerful, efficient solutions that drive performance and reliability.
                </p>
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* Mission & Vision */}
      <section className="section-spacing bg-background">
        <div className="container-custom">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <AnimatedSection animation="fadeRight">
              <img
                src={aboutManufacturingImg}
                alt="Precision manufacturing equipment at Vectis facility"
                className="rounded-lg shadow-xl"
              />
            </AnimatedSection>
            <AnimatedSection animation="fadeLeft" delay={100}>
              <div className="space-y-8">
                <div>
                  <h2 className="text-3xl font-bold mb-4">Our Mission</h2>
                  <p className="text-lg text-muted-foreground leading-relaxed">
                    To deliver exceptional manufacturing solutions that exceed our clients' expectations through precision engineering, innovative technology, and unwavering commitment to quality. We transform concepts into reality, supporting businesses across industries with reliable, scalable manufacturing partnerships.
                  </p>
                </div>
                <div>
                  <h2 className="text-3xl font-bold mb-4">Our Vision</h2>
                  <p className="text-lg text-muted-foreground leading-relaxed">
                    To be the preferred manufacturing partner for companies seeking precision, reliability, and innovation. We envision a future where advanced manufacturing technology and expert craftsmanship combine to solve complex engineering challenges and drive industrial progress.
                  </p>
                </div>
              </div>
            </AnimatedSection>
          </div>
        </div>
      </section>

      {/* Core Values */}
      <section className="section-spacing bg-muted">
        <div className="container-custom">
          <AnimatedSection animation="fadeUp">
            <div className="text-center max-w-3xl mx-auto mb-16">
              <h2 className="mb-4">Our Core Values</h2>
              <p className="text-lg text-muted-foreground">
                The principles that guide every project and partnership we undertake.
              </p>
            </div>
          </AnimatedSection>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {values.map((value, index) => {
              const Icon = value.icon;
              return (
                <AnimatedSection key={index} animation="fadeUp" delay={index * 100}>
                  <Card className="border-2 hover:shadow-lg transition-shadow h-full">
                    <CardContent className="p-6">
                      <div className="w-14 h-14 bg-primary rounded-lg flex items-center justify-center mb-4">
                        <Icon className="h-7 w-7 text-primary-foreground" />
                      </div>
                      <h3 className="text-xl font-bold mb-2">{value.title}</h3>
                      <p className="text-muted-foreground">{value.description}</p>
                    </CardContent>
                  </Card>
                </AnimatedSection>
              );
            })}
          </div>
        </div>
      </section>

      {/* Company Stats */}
      <section className="section-spacing bg-accent text-accent-foreground">
        <div className="container-custom">
          <AnimatedSection animation="fadeUp">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              <div className="text-center">
                <div className="text-5xl font-bold text-primary mb-2">25+</div>
                <div className="text-gray-200">Years in Business</div>
              </div>
              <div className="text-center">
                <div className="text-5xl font-bold text-primary mb-2">500+</div>
                <div className="text-gray-200">Projects Delivered</div>
              </div>
              <div className="text-center">
                <div className="text-5xl font-bold text-primary mb-2">50+</div>
                <div className="text-gray-200">Expert Team Members</div>
              </div>
              <div className="text-center">
                <div className="text-5xl font-bold text-primary mb-2">98%</div>
                <div className="text-gray-200">Client Satisfaction</div>
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* Expertise Section */}
      <section className="section-spacing bg-background">
        <div className="container-custom">
          <AnimatedSection animation="fadeUp">
            <div className="max-w-3xl mx-auto text-center">
              <h2 className="mb-6">Expertise Across Industries</h2>
              <p className="text-lg text-muted-foreground mb-12">
                Our diverse experience spans multiple sectors, giving us unique insights into the specific challenges and requirements of each industry we serve.
              </p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {["Aerospace", "Automotive", "Medical Devices", "Industrial Equipment", "Energy & Power", "Defense & Military"].map((industry, index) => (
                  <AnimatedSection key={industry} animation="scaleIn" delay={index * 50}>
                    <div className="bg-muted rounded-lg p-4 font-semibold text-center hover:bg-primary hover:text-primary-foreground transition-colors">
                      {industry}
                    </div>
                  </AnimatedSection>
                ))}
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default About;