# OMTRA-ForcelabElixir Integration Specification

**Author:** Manus AI  
**Date:** December 9, 2025  
**Version:** 1.0

---

## Executive Summary

This document provides a comprehensive technical specification for integrating **OMTRA** (a multi-task generative model for structure-based drug design) into the **ForcelabElixir** stack. OMTRA is a flow-matching based generative model that supports various tasks relevant to structure-based drug design, including unconditional 3D de novo molecule generation, protein pocket-conditioned molecule design, protein-ligand docking, conformer generation, and pharmacophore-conditioned molecule generation.

The integration will create a unified web platform that combines OMTRA's powerful AI-driven molecular generation capabilities with a modern full-stack architecture featuring user authentication, database persistence, job queue management, and an interactive web interface.

---

## Table of Contents

1. [OMTRA Overview](#omtra-overview)
2. [Architecture Design](#architecture-design)
3. [Technology Stack](#technology-stack)
4. [System Components](#system-components)
5. [Database Schema](#database-schema)
6. [API Endpoints](#api-endpoints)
7. [Frontend Components](#frontend-components)
8. [Integration Steps](#integration-steps)
9. [Configuration Requirements](#configuration-requirements)
10. [Deployment Strategy](#deployment-strategy)
11. [Testing Requirements](#testing-requirements)
12. [References](#references)

---

## OMTRA Overview

### Capabilities

OMTRA supports the following drug design tasks:

| Task Category | Task Name | Description |
|--------------|-----------|-------------|
| **Unconditional Generation** | `denovo_ligand_condensed` | Generate novel drug-like molecules from scratch |
| **Protein-Conditioned** | `fixed_protein_ligand_denovo_condensed` | Design ligands for a fixed protein binding site |
| **Protein-Conditioned** | `protein_ligand_denovo_condensed` | Joint generation of ligand with flexible protein |
| **Docking** | `rigid_docking_condensed` | Dock a known ligand into a fixed protein structure |
| **Docking** | `flexible_docking_condensed` | Dock with protein flexibility |
| **Docking** | `expapo_conditioned_ligand_docking_condensed` | Docking from experimental apo structure |
| **Docking** | `predapo_conditioned_ligand_docking_condensed` | Docking from predicted apo structure |
| **Conformer Generation** | `ligand_conformer_condensed` | Generate 3D conformations for a given ligand |
| **Pharmacophore-Conditioned** | `denovo_ligand_pharmacophore_condensed` | Generate ligand and pharmacophore jointly |
| **Pharmacophore-Conditioned** | `denovo_ligand_from_pharmacophore_condensed` | Design ligand matching a given pharmacophore |
| **Pharmacophore-Conditioned** | `ligand_conformer_from_pharmacophore_condensed` | Generate conformer satisfying pharmacophore |
| **Pharmacophore-Conditioned** | `fixed_protein_pharmacophore_ligand_denovo_condensed` | Design ligand for protein with pharmacophore constraints |
| **Pharmacophore-Conditioned** | `rigid_docking_pharmacophore_condensed` | Dock ligand with pharmacophore constraints |

### Core Arguments

OMTRA's CLI accepts the following core arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | string | *required* | The sampling task to perform |
| `--checkpoint` | path | auto | Path to model checkpoint (auto-detected from task) |
| `--n_samples` | int | 100 | Number of samples to generate |
| `--n_timesteps` | int | 250 | Number of integration steps during sampling |
| `--output_dir` | path | None | Directory to save output files |
| `--metrics` | flag | False | Compute evaluation metrics on generated samples |
| `--protein_file` | path | None | Protein structure file (PDB or CIF format) |
| `--ligand_file` | path | None | Ligand structure file (SDF format) |
| `--pharmacophore_file` | path | None | Pharmacophore constraints file (XYZ format) |

---

## Architecture Design

### High-Level Architecture

The ForcelabElixir platform will follow a **microservices-inspired architecture** with the following layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React/Next.js)                │
│  - Job Submission Interface                                 │
│  - 3D Molecular Viewer (3Dmol.js)                          │
│  - Job Status Dashboard                                     │
│  - User Authentication                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                  Backend API (Node.js/Express)              │
│  - Authentication & Authorization                           │
│  - Job Management                                           │
│  - File Upload/Download                                     │
│  - Database Operations                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Job Queue (Redis + RQ)                   │
│  - Task Queueing                                           │
│  - Job Status Tracking                                      │
│  - Result Storage                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Worker Service (Python)                    │
│  - OMTRA CLI Integration                                   │
│  - Molecular Generation                                     │
│  - Result Processing                                        │
│  - Metrics Computation                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Database (PostgreSQL)                    │
│  - User Data                                               │
│  - Job Records                                             │
│  - Generated Molecules                                      │
│  - Metrics & Analytics                                      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

**Job Submission Flow:**

1. User uploads input files (protein PDB, ligand SDF, pharmacophore XYZ) through the web interface
2. Frontend validates file types and sizes
3. Backend API receives files, stores them in S3, and creates a job record in the database
4. Job parameters are serialized and enqueued to Redis queue
5. Worker picks up the job, invokes OMTRA CLI with appropriate parameters
6. OMTRA generates molecules and saves outputs to the job directory
7. Worker processes results, extracts metrics, and updates job status in database
8. Frontend polls for job status updates and displays results when complete

---

## Technology Stack

### Frontend

- **Framework:** React 18 with Next.js 14 (App Router)
- **Styling:** Tailwind CSS 4 with custom design system
- **3D Visualization:** 3Dmol.js for molecular structure rendering
- **State Management:** React Context API + SWR for data fetching
- **File Upload:** React Dropzone with chunked upload support
- **Charts:** Recharts for metrics visualization

### Backend

- **Runtime:** Node.js 22 with TypeScript
- **Framework:** Express.js or Fastify
- **ORM:** Drizzle ORM with PostgreSQL
- **Authentication:** JWT-based auth with OAuth2 support
- **File Storage:** AWS S3 (via built-in storage helpers)
- **Job Queue:** Redis with Bull or BullMQ

### Worker Service

- **Language:** Python 3.11
- **Queue Client:** RQ (Redis Queue) or Celery
- **OMTRA Integration:** Direct CLI invocation via subprocess
- **Dependencies:** PyTorch 2.4, DGL 2.4, RDKit, and OMTRA package

### Database

- **Primary Database:** PostgreSQL 15+
- **Schema Management:** Drizzle Kit migrations
- **Connection Pooling:** Built-in connection pool

### Infrastructure

- **Containerization:** Docker with multi-stage builds
- **Orchestration:** Docker Compose for local development
- **Model Storage:** Persistent volume for OMTRA checkpoints (~2-5GB)
- **File Storage:** S3-compatible object storage

---

## System Components

### 1. Frontend Application

**Location:** `/home/ubuntu/forcelab-elixir/src/`

**Key Pages:**

- **Home/Dashboard** (`/`): Overview of recent jobs, quick start guide
- **New Job** (`/jobs/new`): Job submission form with file upload
- **Job List** (`/jobs`): Paginated list of user's jobs with status
- **Job Details** (`/jobs/[id]`): Detailed view with 3D viewer, metrics, download
- **Documentation** (`/docs`): API documentation and usage examples

**Key Components:**

```typescript
// Job Submission Form
interface JobSubmissionFormProps {
  onSubmit: (params: JobParams) => Promise<void>;
}

interface JobParams {
  task: string;
  n_samples: number;
  n_timesteps: number;
  protein_file?: File;
  ligand_file?: File;
  pharmacophore_file?: File;
  use_gt_n_lig_atoms?: boolean;
  n_lig_atom_margin?: number;
  stochastic_sampling?: boolean;
  compute_metrics?: boolean;
}

// Molecular Viewer Component
interface MolecularViewerProps {
  moleculeData: string; // SDF format
  proteinData?: string; // PDB format
  pharmacophoreData?: string; // XYZ format
  viewerStyle?: ViewerStyle;
}

// Job Status Component
interface JobStatusProps {
  jobId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  error?: string;
}
```

### 2. Backend API

**Location:** `/home/ubuntu/forcelab-elixir/server/`

**Core Modules:**

```typescript
// Job Management Service
class JobService {
  async createJob(userId: string, params: JobParams): Promise<Job>;
  async getJob(jobId: string): Promise<Job>;
  async listJobs(userId: string, filters: JobFilters): Promise<Job[]>;
  async cancelJob(jobId: string): Promise<void>;
  async deleteJob(jobId: string): Promise<void>;
}

// File Management Service
class FileService {
  async uploadFile(file: Buffer, filename: string, jobId: string): Promise<string>;
  async downloadFile(fileKey: string): Promise<Buffer>;
  async deleteFiles(jobId: string): Promise<void>;
}

// Queue Service
class QueueService {
  async enqueueJob(jobId: string, params: JobParams): Promise<string>;
  async getJobStatus(jobId: string): Promise<JobStatus>;
  async cancelJob(jobId: string): Promise<void>;
}
```

### 3. Worker Service

**Location:** `/home/ubuntu/forcelab-elixir/worker/`

**Core Functions:**

```python
# worker.py
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional

class OMTRAWorker:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
    
    def process_job(self, job_id: str, params: Dict) -> Dict:
        """
        Process a single OMTRA job.
        
        Args:
            job_id: Unique job identifier
            params: Job parameters including task, files, and settings
            
        Returns:
            Dictionary containing job results and metrics
        """
        # Create job directory
        job_dir = Path(f"/tmp/omtra_jobs/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Download input files from S3
        input_files = self._download_input_files(params, job_dir)
        
        # Build OMTRA command
        cmd = self._build_omtra_command(params, input_files, job_dir)
        
        # Execute OMTRA
        result = self._execute_omtra(cmd)
        
        # Process outputs
        outputs = self._process_outputs(job_dir)
        
        # Upload results to S3
        result_urls = self._upload_results(job_id, outputs)
        
        # Compute metrics if requested
        metrics = None
        if params.get('compute_metrics'):
            metrics = self._compute_metrics(outputs)
        
        return {
            'status': 'completed',
            'outputs': result_urls,
            'metrics': metrics,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    
    def _build_omtra_command(self, params: Dict, input_files: Dict, output_dir: Path) -> List[str]:
        """Build OMTRA CLI command from parameters."""
        cmd = ['omtra', '--task', params['task']]
        
        if 'n_samples' in params:
            cmd.extend(['--n_samples', str(params['n_samples'])])
        
        if 'n_timesteps' in params:
            cmd.extend(['--n_timesteps', str(params['n_timesteps'])])
        
        if 'protein_file' in input_files:
            cmd.extend(['--protein_file', str(input_files['protein_file'])])
        
        if 'ligand_file' in input_files:
            cmd.extend(['--ligand_file', str(input_files['ligand_file'])])
        
        if 'pharmacophore_file' in input_files:
            cmd.extend(['--pharmacophore_file', str(input_files['pharmacophore_file'])])
        
        cmd.extend(['--output_dir', str(output_dir)])
        
        if params.get('compute_metrics'):
            cmd.append('--metrics')
        
        if params.get('stochastic_sampling'):
            cmd.append('--stochastic_sampling')
        
        return cmd
```

---

## Database Schema

### Users Table

```sql
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  open_id VARCHAR(255) UNIQUE NOT NULL,
  email VARCHAR(255),
  name VARCHAR(255),
  avatar_url TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Jobs Table

```sql
CREATE TABLE jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  task VARCHAR(100) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'pending',
  parameters JSONB NOT NULL,
  input_files JSONB,
  output_files JSONB,
  metrics JSONB,
  error_message TEXT,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_user_id (user_id),
  INDEX idx_status (status),
  INDEX idx_created_at (created_at)
);
```

### Molecules Table

```sql
CREATE TABLE molecules (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  smiles TEXT NOT NULL,
  sdf_data TEXT,
  properties JSONB,
  metrics JSONB,
  rank INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_job_id (job_id),
  INDEX idx_smiles (smiles)
);
```

### Job Files Table

```sql
CREATE TABLE job_files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  file_type VARCHAR(50) NOT NULL,
  file_name VARCHAR(255) NOT NULL,
  file_key TEXT NOT NULL,
  file_size BIGINT,
  mime_type VARCHAR(100),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_job_id (job_id),
  INDEX idx_file_type (file_type)
);
```

---

## API Endpoints

### Authentication

```
POST   /api/auth/login          - User login
POST   /api/auth/logout         - User logout
GET    /api/auth/me             - Get current user
```

### Jobs

```
POST   /api/jobs                - Create new job
GET    /api/jobs                - List user's jobs (paginated)
GET    /api/jobs/:id            - Get job details
DELETE /api/jobs/:id            - Delete job
POST   /api/jobs/:id/cancel     - Cancel running job
```

### Files

```
POST   /api/files/upload        - Upload input file
GET    /api/files/:key          - Download file
DELETE /api/files/:key          - Delete file
```

### Results

```
GET    /api/jobs/:id/results    - Get job results
GET    /api/jobs/:id/molecules  - List generated molecules
GET    /api/jobs/:id/metrics    - Get job metrics
GET    /api/jobs/:id/download   - Download all results as ZIP
```

### System

```
GET    /api/health              - Health check
GET    /api/tasks               - List available OMTRA tasks
GET    /api/checkpoints         - List available model checkpoints
```

---

## Frontend Components

### Component Structure

```
src/
├── components/
│   ├── jobs/
│   │   ├── JobSubmissionForm.tsx
│   │   ├── JobList.tsx
│   │   ├── JobCard.tsx
│   │   ├── JobDetails.tsx
│   │   └── JobStatusBadge.tsx
│   ├── molecules/
│   │   ├── MolecularViewer.tsx
│   │   ├── MoleculeCard.tsx
│   │   ├── MoleculeList.tsx
│   │   └── PharmacophoreViewer.tsx
│   ├── upload/
│   │   ├── FileUpload.tsx
│   │   ├── FilePreview.tsx
│   │   └── UploadProgress.tsx
│   ├── metrics/
│   │   ├── MetricsTable.tsx
│   │   ├── MetricsChart.tsx
│   │   └── MetricsSummary.tsx
│   └── common/
│       ├── Header.tsx
│       ├── Footer.tsx
│       ├── Sidebar.tsx
│       └── LoadingSpinner.tsx
```

### Key Component Specifications

**JobSubmissionForm.tsx:**
- Task selection dropdown with descriptions
- Conditional file upload fields based on selected task
- Parameter configuration (n_samples, n_timesteps, etc.)
- Advanced options collapsible section
- Form validation with error messages
- Submit button with loading state

**MolecularViewer.tsx:**
- Integration with 3Dmol.js library
- Support for multiple molecule formats (SDF, PDB)
- Interactive controls (rotate, zoom, pan)
- Style customization (stick, cartoon, surface)
- Pharmacophore feature visualization
- Screenshot/export functionality

**JobList.tsx:**
- Paginated table/grid of jobs
- Status badges with color coding
- Quick actions (view, cancel, delete)
- Filtering by status and task type
- Sorting by date, status
- Search functionality

---

## Integration Steps

### Phase 1: Project Initialization

1. **Initialize ForcelabElixir project:**
   ```bash
   cd /home/ubuntu
   # Use webdev_init_project tool with web-db-user features
   ```

2. **Create todo.md with all planned features:**
   ```markdown
   # ForcelabElixir TODO
   
   ## Phase 1: Setup
   - [ ] Initialize project structure
   - [ ] Set up database schema
   - [ ] Configure S3 storage
   - [ ] Set up Redis for job queue
   
   ## Phase 2: OMTRA Integration
   - [ ] Install OMTRA dependencies in worker container
   - [ ] Download model checkpoints
   - [ ] Create Python worker service
   - [ ] Test OMTRA CLI integration
   
   ## Phase 3: Backend API
   - [ ] Implement job creation endpoint
   - [ ] Implement file upload/download
   - [ ] Implement job queue integration
   - [ ] Implement job status polling
   
   ## Phase 4: Frontend
   - [ ] Create job submission form
   - [ ] Implement 3D molecular viewer
   - [ ] Create job list and details pages
   - [ ] Add metrics visualization
   
   ## Phase 5: Testing & Deployment
   - [ ] Write unit tests for API
   - [ ] Write integration tests
   - [ ] Create documentation
   - [ ] Deploy and test
   ```

### Phase 2: OMTRA Setup

1. **Install OMTRA in worker environment:**
   ```bash
   # Create worker Dockerfile
   FROM python:3.11-slim
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       git \
       wget \
       && rm -rf /var/lib/apt/lists/*
   
   # Install CUDA-enabled PyTorch and dependencies
   RUN pip install uv
   COPY requirements-cuda.txt .
   RUN uv pip install -r requirements-cuda.txt --system
   
   # Install OMTRA
   COPY OMTRA /srv/OMTRA
   WORKDIR /srv/OMTRA
   RUN uv pip install -e . --system
   
   # Download model checkpoints
   RUN mkdir -p /srv/checkpoints && \
       wget -r -np -nH --cut-dirs=3 -R "index.html*" \
       -P /srv/checkpoints \
       https://bits.csb.pitt.edu/files/OMTRA/omtra_v0_weights/
   
   WORKDIR /srv/worker
   COPY worker.py .
   CMD ["python", "worker.py"]
   ```

2. **Create worker service:**
   - Implement job processing logic
   - Add error handling and logging
   - Implement result upload to S3
   - Add metrics computation

### Phase 3: Database Schema

1. **Create Drizzle schema files:**
   ```typescript
   // server/db/schema/users.ts
   export const users = pgTable('users', {
     id: uuid('id').primaryKey().defaultRandom(),
     openId: varchar('open_id', { length: 255 }).unique().notNull(),
     email: varchar('email', { length: 255 }),
     name: varchar('name', { length: 255 }),
     avatarUrl: text('avatar_url'),
     createdAt: timestamp('created_at').defaultNow(),
     updatedAt: timestamp('updated_at').defaultNow(),
   });
   
   // server/db/schema/jobs.ts
   export const jobs = pgTable('jobs', {
     id: uuid('id').primaryKey().defaultRandom(),
     userId: uuid('user_id').references(() => users.id).notNull(),
     task: varchar('task', { length: 100 }).notNull(),
     status: varchar('status', { length: 20 }).default('pending').notNull(),
     parameters: jsonb('parameters').notNull(),
     inputFiles: jsonb('input_files'),
     outputFiles: jsonb('output_files'),
     metrics: jsonb('metrics'),
     errorMessage: text('error_message'),
     startedAt: timestamp('started_at'),
     completedAt: timestamp('completed_at'),
     createdAt: timestamp('created_at').defaultNow(),
     updatedAt: timestamp('updated_at').defaultNow(),
   }, (table) => ({
     userIdIdx: index('idx_user_id').on(table.userId),
     statusIdx: index('idx_status').on(table.status),
     createdAtIdx: index('idx_created_at').on(table.createdAt),
   }));
   ```

2. **Run migrations:**
   ```bash
   cd /home/ubuntu/forcelab-elixir
   pnpm db:push
   ```

### Phase 4: Backend Implementation

1. **Create job service:**
   - Implement CRUD operations for jobs
   - Add job queue integration
   - Implement file upload/download
   - Add authentication middleware

2. **Create API routes:**
   - Implement all endpoints listed in API Endpoints section
   - Add request validation
   - Add error handling
   - Add rate limiting

### Phase 5: Frontend Implementation

1. **Create pages:**
   - Home/Dashboard
   - New Job form
   - Job list
   - Job details

2. **Implement components:**
   - Job submission form with file upload
   - 3D molecular viewer using 3Dmol.js
   - Job status dashboard
   - Metrics visualization

3. **Add state management:**
   - Use SWR for data fetching
   - Implement real-time job status updates
   - Add optimistic UI updates

### Phase 6: Testing

1. **Write unit tests:**
   ```typescript
   // Example test for job creation
   describe('Job Creation', () => {
     it('should create a new job with valid parameters', async () => {
       const params = {
         task: 'denovo_ligand_condensed',
         n_samples: 10,
         n_timesteps: 250,
       };
       
       const response = await request(app)
         .post('/api/jobs')
         .send(params)
         .expect(201);
       
       expect(response.body).toHaveProperty('id');
       expect(response.body.status).toBe('pending');
     });
   });
   ```

2. **Write integration tests:**
   - Test complete job workflow
   - Test file upload and download
   - Test authentication flow

---

## Configuration Requirements

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/forcelab_elixir

# Redis
REDIS_URL=redis://localhost:6379

# S3 Storage (auto-configured by Manus)
# BUILT_IN_FORGE_API_KEY
# BUILT_IN_FORGE_API_URL

# Authentication (auto-configured by Manus)
# JWT_SECRET
# OAUTH_SERVER_URL
# OWNER_NAME
# OWNER_OPEN_ID

# OMTRA Configuration
CHECKPOINT_DIR=/srv/checkpoints
MAX_FILE_SIZE=26214400
MAX_FILES_PER_JOB=3
JOB_TTL_HOURS=48

# Worker Configuration
WORKER_CONCURRENCY=2
WORKER_TIMEOUT=600
```

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: forcelab_elixir
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  worker:
    build:
      context: .
      dockerfile: worker/Dockerfile
    depends_on:
      - redis
      - postgres
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/forcelab_elixir
      - CHECKPOINT_DIR=/srv/checkpoints
    volumes:
      - checkpoint_data:/srv/checkpoints
      - ./worker:/srv/worker
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  postgres_data:
  checkpoint_data:
```

---

## Deployment Strategy

### Local Development

1. Start services with Docker Compose:
   ```bash
   docker-compose up -d postgres redis worker
   ```

2. Run database migrations:
   ```bash
   pnpm db:push
   ```

3. Start development server:
   ```bash
   pnpm dev
   ```

### Production Deployment

1. **Build optimized images:**
   ```bash
   docker build -t forcelab-elixir:latest .
   docker build -t forcelab-elixir-worker:latest -f worker/Dockerfile .
   ```

2. **Deploy to Manus hosting:**
   - Create checkpoint in Manus UI
   - Click Publish button
   - Configure custom domain if needed

3. **Post-deployment tasks:**
   - Verify model checkpoints are accessible
   - Test job submission and processing
   - Monitor worker logs for errors
   - Set up monitoring and alerts

---

## Testing Requirements

### Unit Tests

- **Backend Services:**
  - Job service CRUD operations
  - File upload/download
  - Authentication middleware
  - Queue service operations

- **Frontend Components:**
  - Job submission form validation
  - File upload component
  - Molecular viewer rendering
  - Job status updates

### Integration Tests

- **End-to-End Job Flow:**
  1. User creates account and logs in
  2. User submits job with input files
  3. Job is queued and processed by worker
  4. Results are stored and displayed
  5. User downloads results

- **OMTRA Integration:**
  - Test each task type with sample inputs
  - Verify output file formats
  - Validate metrics computation
  - Test error handling for invalid inputs

### Performance Tests

- **Load Testing:**
  - Concurrent job submissions
  - Large file uploads
  - Multiple users accessing results

- **Scalability Testing:**
  - Worker scaling under load
  - Database query performance
  - S3 upload/download throughput

---

## References

[1] OMTRA GitHub Repository: https://github.com/gnina/OMTRA  
[2] OMTRA Preprint: https://arxiv.org/abs/2512.05080  
[3] Manus Web Development Documentation: https://docs.manus.im/webdev  
[4] 3Dmol.js Documentation: https://3dmol.csb.pitt.edu/  
[5] RDKit Documentation: https://www.rdkit.org/docs/  
[6] Drizzle ORM Documentation: https://orm.drizzle.team/  
[7] Next.js Documentation: https://nextjs.org/docs  

---

## Appendix A: OMTRA Task Descriptions

### Unconditional Generation Tasks

**denovo_ligand_condensed:** Generates novel drug-like molecules from scratch without any conditioning. This task is useful for exploring chemical space and discovering novel molecular scaffolds.

### Protein-Conditioned Tasks

**fixed_protein_ligand_denovo_condensed:** Designs ligands for a fixed protein binding site. The protein structure remains rigid during generation, and the model generates ligands that fit the binding pocket.

**protein_ligand_denovo_condensed:** Joint generation of ligand with flexible protein. The protein structure can adapt during generation, allowing for induced-fit scenarios.

### Docking Tasks

**rigid_docking_condensed:** Docks a known ligand into a fixed protein structure. Both protein and ligand remain rigid during the docking process.

**flexible_docking_condensed:** Docks a ligand with protein flexibility. The protein side chains can adapt to accommodate the ligand.

**expapo_conditioned_ligand_docking_condensed:** Docking starting from an experimental apo (ligand-free) protein structure.

**predapo_conditioned_ligand_docking_condensed:** Docking starting from a predicted apo structure.

### Conformer Generation Tasks

**ligand_conformer_condensed:** Generates multiple 3D conformations for a given ligand. This is useful for exploring the conformational space of flexible molecules.

### Pharmacophore-Conditioned Tasks

**denovo_ligand_pharmacophore_condensed:** Generates both ligand and pharmacophore jointly. The model learns to generate molecules along with their key pharmacophoric features.

**denovo_ligand_from_pharmacophore_condensed:** Designs ligands that match a given pharmacophore. The pharmacophore defines spatial constraints that the generated molecule must satisfy.

**ligand_conformer_from_pharmacophore_condensed:** Generates conformers that satisfy pharmacophore constraints.

**fixed_protein_pharmacophore_ligand_denovo_condensed:** Designs ligands for a protein binding site with additional pharmacophore constraints.

**rigid_docking_pharmacophore_condensed:** Docks ligands with pharmacophore constraints, ensuring the docked pose satisfies the specified pharmacophoric features.

---

## Appendix B: File Format Specifications

### Protein Files (PDB/CIF)

OMTRA accepts protein structures in standard PDB or mmCIF format. The protein file should contain:
- Atomic coordinates for all atoms
- Residue information
- Chain identifiers
- Optional: B-factors and occupancy values

### Ligand Files (SDF)

Ligand structures should be provided in SDF (Structure Data File) format with:
- 3D atomic coordinates
- Bond information
- Optional: molecular properties and annotations

### Pharmacophore Files (XYZ)

Pharmacophore constraints are specified in XYZ format with the following structure:
```
<number_of_features>
<comment_line>
<feature_type> <x> <y> <z> <radius>
...
```

Supported feature types:
- Hydrophobic
- Aromatic
- HBond Acceptor
- HBond Donor
- Positive Ionizable
- Negative Ionizable

---

## Appendix C: Metrics Computed by OMTRA

When the `--metrics` flag is enabled, OMTRA computes the following metrics for generated molecules:

### Molecular Properties

- **Molecular Weight:** Total molecular weight in Daltons
- **LogP:** Octanol-water partition coefficient (lipophilicity)
- **TPSA:** Topological polar surface area
- **HBD/HBA:** Number of hydrogen bond donors and acceptors
- **Rotatable Bonds:** Number of rotatable bonds (flexibility)
- **Aromatic Rings:** Number of aromatic ring systems

### Drug-likeness Metrics

- **Lipinski's Rule of Five:** Pass/fail for each criterion
- **QED:** Quantitative Estimate of Drug-likeness (0-1 scale)
- **Synthetic Accessibility:** Ease of synthesis score (1-10 scale)

### Structural Metrics

- **RMSD:** Root-mean-square deviation from reference (if applicable)
- **Strain Energy:** Molecular strain energy
- **Validity:** Chemical validity check (valid bonds, valences, etc.)

### Binding Metrics (for docking tasks)

- **Binding Affinity:** Predicted binding affinity (kcal/mol)
- **Interaction Fingerprint:** Protein-ligand interaction profile
- **Buried Surface Area:** Surface area buried upon binding

---

**End of Document**
