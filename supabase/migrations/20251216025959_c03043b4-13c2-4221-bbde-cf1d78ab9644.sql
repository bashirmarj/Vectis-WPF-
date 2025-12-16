-- Create status enum for project parts
CREATE TYPE public.part_status AS ENUM ('draft', 'confirmation_required', 'quoted', 'order_preparation', 'completed');

-- Create customer_projects table
CREATE TABLE public.customer_projects (
    id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    is_shared BOOLEAN DEFAULT false,
    parent_folder_id UUID REFERENCES public.customer_projects(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create project_parts table
CREATE TABLE public.project_parts (
    id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES public.customer_projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    thumbnail_url TEXT,
    part_number TEXT,
    processing_method TEXT DEFAULT 'CNC-Milling',
    material TEXT DEFAULT 'Aluminum 6061',
    surface_treatment TEXT,
    status part_status NOT NULL DEFAULT 'draft',
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price NUMERIC,
    subtotal NUMERIC,
    mesh_id UUID REFERENCES public.cad_meshes(id) ON DELETE SET NULL,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.customer_projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.project_parts ENABLE ROW LEVEL SECURITY;

-- RLS Policies for customer_projects
CREATE POLICY "Users can view their own projects"
ON public.customer_projects FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own projects"
ON public.customer_projects FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own projects"
ON public.customer_projects FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own projects"
ON public.customer_projects FOR DELETE
USING (auth.uid() = user_id);

-- Admins can view all projects
CREATE POLICY "Admins can view all projects"
ON public.customer_projects FOR SELECT
USING (has_role(auth.uid(), 'admin'::app_role));

-- RLS Policies for project_parts
CREATE POLICY "Users can view their own parts"
ON public.project_parts FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own parts"
ON public.project_parts FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own parts"
ON public.project_parts FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own parts"
ON public.project_parts FOR DELETE
USING (auth.uid() = user_id);

-- Admins can manage all parts
CREATE POLICY "Admins can view all parts"
ON public.project_parts FOR SELECT
USING (has_role(auth.uid(), 'admin'::app_role));

CREATE POLICY "Admins can update all parts"
ON public.project_parts FOR UPDATE
USING (has_role(auth.uid(), 'admin'::app_role));

-- Create updated_at triggers
CREATE TRIGGER update_customer_projects_updated_at
BEFORE UPDATE ON public.customer_projects
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_project_parts_updated_at
BEFORE UPDATE ON public.project_parts
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

-- Create indexes for better performance
CREATE INDEX idx_customer_projects_user_id ON public.customer_projects(user_id);
CREATE INDEX idx_customer_projects_parent ON public.customer_projects(parent_folder_id);
CREATE INDEX idx_project_parts_project_id ON public.project_parts(project_id);
CREATE INDEX idx_project_parts_user_id ON public.project_parts(user_id);
CREATE INDEX idx_project_parts_status ON public.project_parts(status);