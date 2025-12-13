# OMTRA-ForcelabElixir Quick Start Guide for ChatGPT Codex

**Author:** Manus AI  
**Date:** December 9, 2025  
**Purpose:** Rapid implementation guide for ChatGPT Codex

---

## Overview

This guide provides the essential information needed to quickly implement OMTRA integration into ForcelabElixir using ChatGPT Codex. It includes code templates, configuration snippets, and key integration points.

---

## Project Context

**Project Name:** ForcelabElixir  
**Location:** `/home/ubuntu/forcelab-elixir`  
**OMTRA Location:** `/home/ubuntu/OMTRA`  
**Features:** web-db-user (Full-stack with database and authentication)  
**Stack:** React/Next.js 14, Node.js 22, PostgreSQL, Redis, Python 3.11

---

## Quick Setup Commands

### 1. Initialize Project (If Not Already Done)

```bash
# Project should be initialized via Manus webdev tools
# Project name: forcelab-elixir
# Features: web-db-user
```

### 2. Install Additional Dependencies

```bash
cd /home/ubuntu/forcelab-elixir

# Frontend dependencies
pnpm add 3dmol recharts react-dropzone
pnpm add -D @types/3dmol

# Backend dependencies
pnpm add ioredis bull multer zod
pnpm add -D @types/bull @types/multer
```

### 3. Set Up Redis

```bash
# Redis should be available via Docker Compose or external service
# Set REDIS_URL environment variable
```

### 4. Download OMTRA Model Checkpoints

```bash
cd /home/ubuntu/OMTRA
mkdir -p omtra/trained_models
wget -r -np -nH --cut-dirs=3 -R "index.html*" -P omtra/trained_models https://bits.csb.pitt.edu/files/OMTRA/omtra_v0_weights/
```

---

## Database Schema (Drizzle ORM)

### Jobs Table

```typescript
// server/db/schema/jobs.ts
import { pgTable, uuid, varchar, text, jsonb, timestamp, index } from 'drizzle-orm/pg-core';
import { users } from './users';

export const jobs = pgTable('jobs', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id, { onDelete: 'cascade' }).notNull(),
  task: varchar('task', { length: 100 }).notNull(),
  status: varchar('status', { length: 20 }).default('pending').notNull(),
  parameters: jsonb('parameters').notNull(),
  inputFiles: jsonb('input_files'),
  outputFiles: jsonb('output_files'),
  metrics: jsonb('metrics'),
  errorMessage: text('error_message'),
  startedAt: timestamp('started_at'),
  completedAt: timestamp('completed_at'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
}, (table) => ({
  userIdIdx: index('idx_jobs_user_id').on(table.userId),
  statusIdx: index('idx_jobs_status').on(table.status),
  createdAtIdx: index('idx_jobs_created_at').on(table.createdAt),
}));
```

### Molecules Table

```typescript
// server/db/schema/molecules.ts
import { pgTable, uuid, text, jsonb, integer, timestamp, index } from 'drizzle-orm/pg-core';
import { jobs } from './jobs';

export const molecules = pgTable('molecules', {
  id: uuid('id').primaryKey().defaultRandom(),
  jobId: uuid('job_id').references(() => jobs.id, { onDelete: 'cascade' }).notNull(),
  smiles: text('smiles').notNull(),
  sdfData: text('sdf_data'),
  properties: jsonb('properties'),
  metrics: jsonb('metrics'),
  rank: integer('rank'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
}, (table) => ({
  jobIdIdx: index('idx_molecules_job_id').on(table.jobId),
}));
```

### Job Files Table

```typescript
// server/db/schema/job-files.ts
import { pgTable, uuid, varchar, text, bigint, timestamp, index } from 'drizzle-orm/pg-core';
import { jobs } from './jobs';

export const jobFiles = pgTable('job_files', {
  id: uuid('id').primaryKey().defaultRandom(),
  jobId: uuid('job_id').references(() => jobs.id, { onDelete: 'cascade' }).notNull(),
  fileType: varchar('file_type', { length: 50 }).notNull(),
  fileName: varchar('file_name', { length: 255 }).notNull(),
  fileKey: text('file_key').notNull(),
  fileSize: bigint('file_size', { mode: 'number' }),
  mimeType: varchar('mime_type', { length: 100 }),
  createdAt: timestamp('created_at').defaultNow().notNull(),
}, (table) => ({
  jobIdIdx: index('idx_job_files_job_id').on(table.jobId),
}));
```

**Run Migrations:**
```bash
pnpm db:push
```

---

## Backend Services

### Job Service Template

```typescript
// server/services/job-service.ts
import { db } from '../db';
import { jobs, molecules } from '../db/schema';
import { eq, desc } from 'drizzle-orm';

export class JobService {
  async createJob(userId: string, params: { task: string; parameters: any }) {
    const [job] = await db.insert(jobs).values({
      userId,
      task: params.task,
      status: 'pending',
      parameters: params.parameters,
    }).returning();
    return job;
  }

  async getJob(jobId: string) {
    const [job] = await db.select().from(jobs).where(eq(jobs.id, jobId)).limit(1);
    return job || null;
  }

  async listJobs(userId: string, options?: { status?: string; limit?: number; offset?: number }) {
    const limit = options?.limit || 20;
    const offset = options?.offset || 0;
    
    const results = await db.select()
      .from(jobs)
      .where(eq(jobs.userId, userId))
      .orderBy(desc(jobs.createdAt))
      .limit(limit)
      .offset(offset);
    
    return results;
  }

  async updateJobStatus(jobId: string, status: string, data?: any) {
    const updateData: any = { status, updatedAt: new Date() };
    
    if (status === 'running') updateData.startedAt = new Date();
    if (status === 'completed' || status === 'failed') updateData.completedAt = new Date();
    if (data?.errorMessage) updateData.errorMessage = data.errorMessage;
    if (data?.outputFiles) updateData.outputFiles = data.outputFiles;
    if (data?.metrics) updateData.metrics = data.metrics;
    
    const [job] = await db.update(jobs).set(updateData).where(eq(jobs.id, jobId)).returning();
    return job;
  }

  async deleteJob(jobId: string) {
    await db.delete(jobs).where(eq(jobs.id, jobId));
  }
}

export const jobService = new JobService();
```

### Queue Service Template

```typescript
// server/services/queue-service.ts
import Queue from 'bull';

const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';

export const jobQueue = new Queue('omtra-jobs', redisUrl, {
  defaultJobOptions: {
    attempts: 3,
    backoff: { type: 'exponential', delay: 2000 },
    removeOnComplete: false,
    removeOnFail: false,
  },
});

export class QueueService {
  async enqueueJob(jobId: string, params: any) {
    const job = await jobQueue.add('process-omtra', {
      jobId,
      task: params.task,
      parameters: params.parameters,
      inputFiles: params.inputFiles,
    }, { jobId });
    
    return job.id.toString();
  }

  async getJobStatus(jobId: string) {
    const job = await jobQueue.getJob(jobId);
    if (!job) return null;
    
    const state = await job.getState();
    return { state, progress: job.progress() };
  }

  async cancelJob(jobId: string) {
    const job = await jobQueue.getJob(jobId);
    if (job) await job.remove();
  }
}

export const queueService = new QueueService();
```

---

## API Routes

### Job Routes

```typescript
// server/routes/jobs.ts
import { Router } from 'express';
import { jobService } from '../services/job-service';
import { queueService } from '../services/queue-service';
import { requireAuth } from '../middleware/auth';
import { z } from 'zod';

const router = Router();

const createJobSchema = z.object({
  task: z.string(),
  parameters: z.object({
    n_samples: z.number().int().min(1).max(1000).default(100),
    n_timesteps: z.number().int().min(1).max(1000).default(250),
    compute_metrics: z.boolean().optional(),
  }),
});

router.post('/', requireAuth, async (req, res) => {
  try {
    const validated = createJobSchema.parse(req.body);
    const userId = req.user!.id;
    
    const job = await jobService.createJob(userId, validated);
    await queueService.enqueueJob(job.id, validated);
    
    res.status(201).json(job);
  } catch (error) {
    res.status(400).json({ error: 'Invalid request' });
  }
});

router.get('/', requireAuth, async (req, res) => {
  const userId = req.user!.id;
  const jobs = await jobService.listJobs(userId);
  res.json(jobs);
});

router.get('/:id', requireAuth, async (req, res) => {
  const job = await jobService.getJob(req.params.id);
  if (!job || job.userId !== req.user!.id) {
    return res.status(404).json({ error: 'Job not found' });
  }
  res.json(job);
});

router.delete('/:id', requireAuth, async (req, res) => {
  const job = await jobService.getJob(req.params.id);
  if (!job || job.userId !== req.user!.id) {
    return res.status(404).json({ error: 'Job not found' });
  }
  
  await queueService.cancelJob(req.params.id);
  await jobService.deleteJob(req.params.id);
  
  res.status(204).send();
});

export default router;
```

---

## Worker Service

### Worker Dockerfile

```dockerfile
# worker/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY /home/ubuntu/OMTRA /srv/OMTRA
WORKDIR /srv/OMTRA
RUN uv pip install -r requirements-cuda.txt --system
RUN uv pip install -e . --system

COPY requirements.txt /srv/worker/requirements.txt
WORKDIR /srv/worker
RUN pip install -r requirements.txt

COPY . /srv/worker

RUN mkdir -p /srv/checkpoints /tmp/omtra_jobs

ENV CHECKPOINT_DIR=/srv/checkpoints
ENV PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]
```

### Worker Requirements

```txt
# worker/requirements.txt
redis==5.0.1
rq==1.15.1
boto3==1.34.0
requests==2.31.0
python-dotenv==1.0.0
```

### Worker Main Script

```python
# worker/worker.py
import os
import logging
from pathlib import Path
from redis import Redis
from rq import Worker, Queue, Connection
import subprocess
import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
CHECKPOINT_DIR = Path(os.getenv('CHECKPOINT_DIR', '/srv/checkpoints'))

def process_omtra_job(job_id: str, task: str, parameters: dict, input_files: dict = None):
    """Process an OMTRA job."""
    logger.info(f"Processing job {job_id} with task {task}")
    
    try:
        # Create job directory
        job_dir = Path(f"/tmp/omtra_jobs/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)
        output_dir = job_dir / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Build OMTRA command
        cmd = ['omtra', '--task', task]
        cmd.extend(['--n_samples', str(parameters.get('n_samples', 100))])
        cmd.extend(['--n_timesteps', str(parameters.get('n_timesteps', 250))])
        cmd.extend(['--output_dir', str(output_dir)])
        
        if parameters.get('compute_metrics'):
            cmd.append('--metrics')
        
        # Run OMTRA
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"OMTRA failed: {result.stderr}")
            return {'status': 'failed', 'error': result.stderr}
        
        logger.info(f"Job {job_id} completed successfully")
        return {'status': 'completed', 'output_dir': str(output_dir)}
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
        return {'status': 'failed', 'error': str(e)}

if __name__ == '__main__':
    redis_conn = Redis.from_url(REDIS_URL)
    queue = Queue('omtra-jobs', connection=redis_conn)
    
    logger.info("Starting OMTRA worker...")
    with Connection(redis_conn):
        worker = Worker([queue])
        worker.work()
```

---

## Frontend Components

### Job Submission Form

```typescript
// src/components/jobs/JobSubmissionForm.tsx
'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

const TASKS = [
  { value: 'denovo_ligand_condensed', label: 'De Novo Generation' },
  { value: 'fixed_protein_ligand_denovo_condensed', label: 'Protein-Conditioned Design' },
  { value: 'rigid_docking_condensed', label: 'Rigid Docking' },
  { value: 'ligand_conformer_condensed', label: 'Conformer Generation' },
];

export function JobSubmissionForm() {
  const router = useRouter();
  const [task, setTask] = useState('');
  const [nSamples, setNSamples] = useState(100);
  const [nTimesteps, setNTimesteps] = useState(250);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      const response = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task,
          parameters: { n_samples: nSamples, n_timesteps: nTimesteps },
        }),
      });

      const job = await response.json();
      router.push(`/jobs/${job.id}`);
    } catch (error) {
      console.error('Error creating job:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <label className="block text-sm font-medium mb-2">Task</label>
        <select
          value={task}
          onChange={(e) => setTask(e.target.value)}
          required
          className="w-full px-4 py-2 border rounded-lg"
        >
          <option value="">Select a task...</option>
          {TASKS.map((t) => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-2">Samples</label>
          <input
            type="number"
            min="1"
            max="1000"
            value={nSamples}
            onChange={(e) => setNSamples(parseInt(e.target.value))}
            className="w-full px-4 py-2 border rounded-lg"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-2">Timesteps</label>
          <input
            type="number"
            min="1"
            max="1000"
            value={nTimesteps}
            onChange={(e) => setNTimesteps(parseInt(e.target.value))}
            className="w-full px-4 py-2 border rounded-lg"
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={isSubmitting || !task}
        className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
      >
        {isSubmitting ? 'Submitting...' : 'Submit Job'}
      </button>
    </form>
  );
}
```

### Molecular Viewer

```typescript
// src/components/molecules/MolecularViewer.tsx
'use client';

import { useEffect, useRef } from 'react';
import * as $3Dmol from '3dmol';

interface Props {
  moleculeData: string;
  proteinData?: string;
}

export function MolecularViewer({ moleculeData, proteinData }: Props) {
  const viewerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!viewerRef.current) return;

    const viewer = $3Dmol.createViewer(viewerRef.current, {
      backgroundColor: 'white',
    });

    if (proteinData) {
      viewer.addModel(proteinData, 'pdb');
      viewer.setStyle({}, { cartoon: { color: 'spectrum' } });
    }

    if (moleculeData) {
      const model = viewer.addModel(moleculeData, 'sdf');
      viewer.setStyle({ model }, { stick: { colorscheme: 'default' } });
    }

    viewer.zoomTo();
    viewer.render();

    return () => viewer.clear();
  }, [moleculeData, proteinData]);

  return <div ref={viewerRef} className="w-full h-[600px] border rounded-lg" />;
}
```

---

## Environment Variables

Add to `.env`:

```bash
# Redis
REDIS_URL=redis://localhost:6379

# Worker
CHECKPOINT_DIR=/srv/checkpoints

# S3 (auto-configured by Manus)
# BUILT_IN_FORGE_API_KEY
# BUILT_IN_FORGE_API_URL
```

---

## Docker Compose (Optional for Local Dev)

```yaml
# docker-compose.yml
version: '3.8'

services:
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
    environment:
      - REDIS_URL=redis://redis:6379
      - CHECKPOINT_DIR=/srv/checkpoints
    volumes:
      - ./OMTRA/omtra/trained_models:/srv/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Testing Commands

```bash
# Test job creation
curl -X POST http://localhost:3000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"task":"denovo_ligand_condensed","parameters":{"n_samples":10,"n_timesteps":250}}'

# Test OMTRA CLI directly
cd /home/ubuntu/OMTRA
omtra --task denovo_ligand_condensed --n_samples 5 --output_dir /tmp/test_output
```

---

## Implementation Priority

1. **Database Schema** - Create tables first
2. **Backend Services** - Job and queue services
3. **API Routes** - Job management endpoints
4. **Worker Service** - OMTRA integration
5. **Frontend Components** - Job submission and viewer
6. **Testing** - End-to-end workflow

---

## Key Files to Create

### Backend
- `server/db/schema/jobs.ts`
- `server/db/schema/molecules.ts`
- `server/db/schema/job-files.ts`
- `server/services/job-service.ts`
- `server/services/queue-service.ts`
- `server/routes/jobs.ts`

### Worker
- `worker/Dockerfile`
- `worker/requirements.txt`
- `worker/worker.py`
- `worker/omtra_runner.py`

### Frontend
- `src/components/jobs/JobSubmissionForm.tsx`
- `src/components/molecules/MolecularViewer.tsx`
- `src/app/jobs/new/page.tsx`
- `src/app/jobs/[id]/page.tsx`

---

## Common Issues & Solutions

**Issue:** OMTRA model checkpoints not found  
**Solution:** Download checkpoints to `/home/ubuntu/OMTRA/omtra/trained_models`

**Issue:** Worker can't connect to Redis  
**Solution:** Verify REDIS_URL environment variable

**Issue:** GPU not available in worker  
**Solution:** Add GPU support to Docker Compose configuration

**Issue:** File upload fails  
**Solution:** Check S3 credentials and bucket configuration

---

## Next Steps for ChatGPT Codex

1. Review the full specification document (`OMTRA_ForcelabElixir_Integration_Spec.md`)
2. Follow the detailed checklist (`OMTRA_Implementation_Checklist.md`)
3. Use code templates from this quick start guide
4. Test each component as you build
5. Mark completed tasks in `todo.md`
6. Save checkpoint when complete

---

**End of Quick Start Guide**
