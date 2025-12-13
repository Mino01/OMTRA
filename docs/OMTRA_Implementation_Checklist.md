# OMTRA-ForcelabElixir Implementation Checklist

**Author:** Manus AI  
**Date:** December 9, 2025  
**Purpose:** Step-by-step implementation guide for ChatGPT Codex

---

## Overview

This checklist provides a detailed, sequential guide for implementing the OMTRA integration into ForcelabElixir. Each task is designed to be completed in order, with clear acceptance criteria and code examples.

---

## Phase 1: Project Initialization

### Task 1.1: Initialize ForcelabElixir Project

**Objective:** Create the base ForcelabElixir web application with database and user authentication.

**Steps:**
1. Use Manus webdev tools to initialize project:
   - Project name: `forcelab-elixir`
   - Features: `web-db-user`
   - Title: "ForcelabElixir - AI-Powered Drug Design Platform"

2. Verify project structure is created:
   ```
   /home/ubuntu/forcelab-elixir/
   ├── src/              # Frontend React/Next.js code
   ├── server/           # Backend Node.js/Express code
   ├── public/           # Static assets
   ├── package.json
   └── drizzle.config.ts
   ```

**Acceptance Criteria:**
- [ ] Project directory exists at `/home/ubuntu/forcelab-elixir`
- [ ] Development server starts successfully
- [ ] Database connection is established
- [ ] User authentication works (login/logout)

---

### Task 1.2: Create Project TODO

**Objective:** Document all planned features in todo.md.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/todo.md` with the following content:

```markdown
# ForcelabElixir TODO

## Phase 1: Project Setup
- [ ] Initialize project structure
- [ ] Set up database schema for jobs and molecules
- [ ] Configure environment variables
- [ ] Set up Redis for job queue

## Phase 2: OMTRA Integration
- [ ] Create worker directory structure
- [ ] Create worker Dockerfile with OMTRA installation
- [ ] Download OMTRA model checkpoints
- [ ] Implement Python worker service
- [ ] Test OMTRA CLI integration with sample inputs

## Phase 3: Backend API Development
- [ ] Create database schema (jobs, molecules, job_files tables)
- [ ] Implement job service (create, read, update, delete)
- [ ] Implement file upload service with S3 integration
- [ ] Implement job queue service with Redis
- [ ] Create API endpoints for job management
- [ ] Create API endpoints for file operations
- [ ] Add authentication middleware to protected routes
- [ ] Add request validation and error handling

## Phase 4: Frontend Development
- [ ] Install 3Dmol.js for molecular visualization
- [ ] Create job submission form component
- [ ] Create file upload component with drag-and-drop
- [ ] Create 3D molecular viewer component
- [ ] Create job list page with status badges
- [ ] Create job details page with results
- [ ] Create metrics visualization component
- [ ] Add real-time job status updates
- [ ] Create documentation page

## Phase 5: Worker Service
- [ ] Implement job processing logic
- [ ] Implement OMTRA command builder
- [ ] Implement file download from S3
- [ ] Implement result upload to S3
- [ ] Implement metrics computation
- [ ] Add error handling and logging
- [ ] Add job timeout handling
- [ ] Test worker with all OMTRA task types

## Phase 6: Testing
- [ ] Write unit tests for job service
- [ ] Write unit tests for file service
- [ ] Write integration tests for job workflow
- [ ] Test all OMTRA task types
- [ ] Test file upload/download
- [ ] Test authentication flow
- [ ] Perform load testing

## Phase 7: Documentation & Deployment
- [ ] Create user documentation
- [ ] Create API documentation
- [ ] Add inline code comments
- [ ] Create deployment guide
- [ ] Save checkpoint
- [ ] Deploy to production
```

**Acceptance Criteria:**
- [ ] todo.md file exists at project root
- [ ] All planned features are documented

---

## Phase 2: Database Schema

### Task 2.1: Create Jobs Table Schema

**Objective:** Define the database schema for storing job information.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/server/db/schema/jobs.ts`:

```typescript
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
  progress: varchar('progress', { length: 50 }),
  startedAt: timestamp('started_at'),
  completedAt: timestamp('completed_at'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
}, (table) => ({
  userIdIdx: index('idx_jobs_user_id').on(table.userId),
  statusIdx: index('idx_jobs_status').on(table.status),
  createdAtIdx: index('idx_jobs_created_at').on(table.createdAt),
}));

export type Job = typeof jobs.$inferSelect;
export type NewJob = typeof jobs.$inferInsert;
```

**Acceptance Criteria:**
- [ ] Schema file created
- [ ] All required fields are defined
- [ ] Indexes are created for performance
- [ ] TypeScript types are exported

---

### Task 2.2: Create Molecules Table Schema

**Objective:** Define the schema for storing generated molecules.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/server/db/schema/molecules.ts`:

```typescript
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
  smilesIdx: index('idx_molecules_smiles').on(table.smiles),
}));

export type Molecule = typeof molecules.$inferSelect;
export type NewMolecule = typeof molecules.$inferInsert;
```

**Acceptance Criteria:**
- [ ] Schema file created
- [ ] Foreign key relationship to jobs table
- [ ] Indexes created

---

### Task 2.3: Create Job Files Table Schema

**Objective:** Track uploaded and generated files.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/server/db/schema/job-files.ts`:

```typescript
import { pgTable, uuid, varchar, text, bigint, timestamp, index } from 'drizzle-orm/pg-core';
import { jobs } from './jobs';

export const jobFiles = pgTable('job_files', {
  id: uuid('id').primaryKey().defaultRandom(),
  jobId: uuid('job_id').references(() => jobs.id, { onDelete: 'cascade' }).notNull(),
  fileType: varchar('file_type', { length: 50 }).notNull(), // 'input' or 'output'
  fileName: varchar('file_name', { length: 255 }).notNull(),
  fileKey: text('file_key').notNull(), // S3 key
  fileSize: bigint('file_size', { mode: 'number' }),
  mimeType: varchar('mime_type', { length: 100 }),
  createdAt: timestamp('created_at').defaultNow().notNull(),
}, (table) => ({
  jobIdIdx: index('idx_job_files_job_id').on(table.jobId),
  fileTypeIdx: index('idx_job_files_file_type').on(table.fileType),
}));

export type JobFile = typeof jobFiles.$inferSelect;
export type NewJobFile = typeof jobFiles.$inferInsert;
```

**Acceptance Criteria:**
- [ ] Schema file created
- [ ] File type field for categorization
- [ ] S3 key stored for retrieval

---

### Task 2.4: Run Database Migrations

**Objective:** Apply schema changes to the database.

**Steps:**
1. Export all schemas in `/home/ubuntu/forcelab-elixir/server/db/schema/index.ts`:

```typescript
export * from './users';
export * from './jobs';
export * from './molecules';
export * from './job-files';
```

2. Run migration:
```bash
cd /home/ubuntu/forcelab-elixir
pnpm db:push
```

3. Verify tables are created:
```bash
# Connect to database and list tables
psql $DATABASE_URL -c "\dt"
```

**Acceptance Criteria:**
- [ ] All tables created successfully
- [ ] Foreign key constraints are in place
- [ ] Indexes are created
- [ ] Mark task as [x] in todo.md

---

## Phase 3: Backend Services

### Task 3.1: Create Job Service

**Objective:** Implement business logic for job management.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/server/services/job-service.ts`:

```typescript
import { db } from '../db';
import { jobs, molecules, jobFiles } from '../db/schema';
import { eq, desc, and } from 'drizzle-orm';
import type { Job, NewJob } from '../db/schema/jobs';

export class JobService {
  /**
   * Create a new job
   */
  async createJob(userId: string, params: {
    task: string;
    parameters: Record<string, any>;
  }): Promise<Job> {
    const [job] = await db.insert(jobs).values({
      userId,
      task: params.task,
      status: 'pending',
      parameters: params.parameters,
    }).returning();
    
    return job;
  }

  /**
   * Get job by ID
   */
  async getJob(jobId: string): Promise<Job | null> {
    const [job] = await db.select()
      .from(jobs)
      .where(eq(jobs.id, jobId))
      .limit(1);
    
    return job || null;
  }

  /**
   * List jobs for a user
   */
  async listJobs(userId: string, options?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ jobs: Job[]; total: number }> {
    const limit = options?.limit || 20;
    const offset = options?.offset || 0;
    
    let query = db.select().from(jobs).where(eq(jobs.userId, userId));
    
    if (options?.status) {
      query = query.where(and(
        eq(jobs.userId, userId),
        eq(jobs.status, options.status)
      ));
    }
    
    const results = await query
      .orderBy(desc(jobs.createdAt))
      .limit(limit)
      .offset(offset);
    
    const [{ count }] = await db.select({ count: jobs.id })
      .from(jobs)
      .where(eq(jobs.userId, userId));
    
    return {
      jobs: results,
      total: Number(count),
    };
  }

  /**
   * Update job status
   */
  async updateJobStatus(jobId: string, status: string, data?: {
    errorMessage?: string;
    progress?: string;
    outputFiles?: any;
    metrics?: any;
  }): Promise<Job> {
    const updateData: any = {
      status,
      updatedAt: new Date(),
    };
    
    if (status === 'running' && !data?.startedAt) {
      updateData.startedAt = new Date();
    }
    
    if (status === 'completed' || status === 'failed') {
      updateData.completedAt = new Date();
    }
    
    if (data?.errorMessage) {
      updateData.errorMessage = data.errorMessage;
    }
    
    if (data?.progress) {
      updateData.progress = data.progress;
    }
    
    if (data?.outputFiles) {
      updateData.outputFiles = data.outputFiles;
    }
    
    if (data?.metrics) {
      updateData.metrics = data.metrics;
    }
    
    const [job] = await db.update(jobs)
      .set(updateData)
      .where(eq(jobs.id, jobId))
      .returning();
    
    return job;
  }

  /**
   * Delete job and associated data
   */
  async deleteJob(jobId: string): Promise<void> {
    // Delete molecules (cascade will handle this, but explicit is better)
    await db.delete(molecules).where(eq(molecules.jobId, jobId));
    
    // Delete job files
    await db.delete(jobFiles).where(eq(jobFiles.jobId, jobId));
    
    // Delete job
    await db.delete(jobs).where(eq(jobs.id, jobId));
  }

  /**
   * Get job with molecules
   */
  async getJobWithMolecules(jobId: string): Promise<{
    job: Job;
    molecules: any[];
  } | null> {
    const job = await this.getJob(jobId);
    if (!job) return null;
    
    const mols = await db.select()
      .from(molecules)
      .where(eq(molecules.jobId, jobId))
      .orderBy(molecules.rank);
    
    return {
      job,
      molecules: mols,
    };
  }
}

export const jobService = new JobService();
```

**Acceptance Criteria:**
- [ ] Service class created with all CRUD methods
- [ ] Proper error handling
- [ ] TypeScript types used throughout
- [ ] Database queries optimized with indexes

---

### Task 3.2: Create File Service

**Objective:** Handle file uploads and downloads with S3.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/server/services/file-service.ts`:

```typescript
import { storagePut, storageGet } from '../lib/storage';
import { db } from '../db';
import { jobFiles } from '../db/schema';
import type { NewJobFile } from '../db/schema/job-files';

export class FileService {
  /**
   * Upload file to S3
   */
  async uploadFile(
    jobId: string,
    file: Buffer,
    fileName: string,
    fileType: 'input' | 'output',
    mimeType?: string
  ): Promise<{ key: string; url: string }> {
    // Generate S3 key
    const key = `jobs/${jobId}/${fileType}/${fileName}`;
    
    // Upload to S3
    const result = await storagePut(key, file, mimeType);
    
    // Save file record to database
    await db.insert(jobFiles).values({
      jobId,
      fileType,
      fileName,
      fileKey: result.key,
      fileSize: file.length,
      mimeType: mimeType || 'application/octet-stream',
    });
    
    return result;
  }

  /**
   * Download file from S3
   */
  async downloadFile(fileKey: string): Promise<{ url: string }> {
    const result = await storageGet(fileKey, 3600); // 1 hour expiry
    return result;
  }

  /**
   * Get files for a job
   */
  async getJobFiles(jobId: string, fileType?: 'input' | 'output') {
    let query = db.select().from(jobFiles).where(eq(jobFiles.jobId, jobId));
    
    if (fileType) {
      query = query.where(and(
        eq(jobFiles.jobId, jobId),
        eq(jobFiles.fileType, fileType)
      ));
    }
    
    return await query;
  }

  /**
   * Delete files for a job
   */
  async deleteJobFiles(jobId: string): Promise<void> {
    // Note: S3 files are not deleted, only database records
    // This allows for data retention if needed
    await db.delete(jobFiles).where(eq(jobFiles.jobId, jobId));
  }

  /**
   * Validate file type
   */
  validateFileType(fileName: string, allowedTypes: string[]): boolean {
    const ext = fileName.split('.').pop()?.toLowerCase();
    return ext ? allowedTypes.includes(ext) : false;
  }

  /**
   * Validate file size
   */
  validateFileSize(fileSize: number, maxSize: number = 25 * 1024 * 1024): boolean {
    return fileSize <= maxSize;
  }
}

export const fileService = new FileService();
```

**Acceptance Criteria:**
- [ ] File upload to S3 works
- [ ] File download from S3 works
- [ ] File validation implemented
- [ ] Database records created for files

---

### Task 3.3: Create Queue Service

**Objective:** Integrate Redis queue for job processing.

**Steps:**
1. Install Redis client:
```bash
cd /home/ubuntu/forcelab-elixir
pnpm add ioredis bull
pnpm add -D @types/bull
```

2. Create `/home/ubuntu/forcelab-elixir/server/services/queue-service.ts`:

```typescript
import Queue from 'bull';
import Redis from 'ioredis';

const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';

// Create Redis connection
const redisClient = new Redis(redisUrl);

// Create job queue
export const jobQueue = new Queue('omtra-jobs', redisUrl, {
  defaultJobOptions: {
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 2000,
    },
    removeOnComplete: false,
    removeOnFail: false,
  },
});

export class QueueService {
  /**
   * Enqueue a new job
   */
  async enqueueJob(jobId: string, params: {
    task: string;
    parameters: Record<string, any>;
    inputFiles?: Record<string, string>; // file type -> S3 key
  }): Promise<string> {
    const job = await jobQueue.add('process-omtra', {
      jobId,
      task: params.task,
      parameters: params.parameters,
      inputFiles: params.inputFiles,
    }, {
      jobId, // Use jobId as Bull job ID for easy tracking
    });
    
    return job.id.toString();
  }

  /**
   * Get job status from queue
   */
  async getJobStatus(jobId: string) {
    const job = await jobQueue.getJob(jobId);
    if (!job) return null;
    
    const state = await job.getState();
    const progress = job.progress();
    
    return {
      state,
      progress,
      failedReason: job.failedReason,
      finishedOn: job.finishedOn,
      processedOn: job.processedOn,
    };
  }

  /**
   * Cancel a job
   */
  async cancelJob(jobId: string): Promise<void> {
    const job = await jobQueue.getJob(jobId);
    if (job) {
      await job.remove();
    }
  }

  /**
   * Retry a failed job
   */
  async retryJob(jobId: string): Promise<void> {
    const job = await jobQueue.getJob(jobId);
    if (job) {
      await job.retry();
    }
  }
}

export const queueService = new QueueService();
```

**Acceptance Criteria:**
- [ ] Redis connection established
- [ ] Job queue created
- [ ] Jobs can be enqueued
- [ ] Job status can be retrieved

---

## Phase 4: API Endpoints

### Task 4.1: Create Job API Routes

**Objective:** Implement REST API endpoints for job management.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/server/routes/jobs.ts`:

```typescript
import { Router } from 'express';
import { jobService } from '../services/job-service';
import { queueService } from '../services/queue-service';
import { fileService } from '../services/file-service';
import { requireAuth } from '../middleware/auth';
import { z } from 'zod';

const router = Router();

// Validation schemas
const createJobSchema = z.object({
  task: z.string(),
  parameters: z.object({
    n_samples: z.number().int().min(1).max(1000).default(100),
    n_timesteps: z.number().int().min(1).max(1000).default(250),
    stochastic_sampling: z.boolean().optional(),
    compute_metrics: z.boolean().optional(),
    use_gt_n_lig_atoms: z.boolean().optional(),
    n_lig_atom_margin: z.number().optional(),
  }),
});

/**
 * POST /api/jobs
 * Create a new job
 */
router.post('/', requireAuth, async (req, res) => {
  try {
    const validated = createJobSchema.parse(req.body);
    const userId = req.user!.id;
    
    // Create job record
    const job = await jobService.createJob(userId, {
      task: validated.task,
      parameters: validated.parameters,
    });
    
    // Enqueue job for processing
    await queueService.enqueueJob(job.id, {
      task: validated.task,
      parameters: validated.parameters,
    });
    
    res.status(201).json(job);
  } catch (error) {
    console.error('Error creating job:', error);
    res.status(400).json({ error: 'Invalid request' });
  }
});

/**
 * GET /api/jobs
 * List user's jobs
 */
router.get('/', requireAuth, async (req, res) => {
  try {
    const userId = req.user!.id;
    const status = req.query.status as string | undefined;
    const limit = parseInt(req.query.limit as string) || 20;
    const offset = parseInt(req.query.offset as string) || 0;
    
    const result = await jobService.listJobs(userId, {
      status,
      limit,
      offset,
    });
    
    res.json(result);
  } catch (error) {
    console.error('Error listing jobs:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * GET /api/jobs/:id
 * Get job details
 */
router.get('/:id', requireAuth, async (req, res) => {
  try {
    const jobId = req.params.id;
    const userId = req.user!.id;
    
    const job = await jobService.getJob(jobId);
    
    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }
    
    // Check ownership
    if (job.userId !== userId) {
      return res.status(403).json({ error: 'Forbidden' });
    }
    
    res.json(job);
  } catch (error) {
    console.error('Error getting job:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * DELETE /api/jobs/:id
 * Delete a job
 */
router.delete('/:id', requireAuth, async (req, res) => {
  try {
    const jobId = req.params.id;
    const userId = req.user!.id;
    
    const job = await jobService.getJob(jobId);
    
    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }
    
    if (job.userId !== userId) {
      return res.status(403).json({ error: 'Forbidden' });
    }
    
    // Cancel job in queue if running
    await queueService.cancelJob(jobId);
    
    // Delete job and associated data
    await jobService.deleteJob(jobId);
    
    res.status(204).send();
  } catch (error) {
    console.error('Error deleting job:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * POST /api/jobs/:id/cancel
 * Cancel a running job
 */
router.post('/:id/cancel', requireAuth, async (req, res) => {
  try {
    const jobId = req.params.id;
    const userId = req.user!.id;
    
    const job = await jobService.getJob(jobId);
    
    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }
    
    if (job.userId !== userId) {
      return res.status(403).json({ error: 'Forbidden' });
    }
    
    // Cancel job in queue
    await queueService.cancelJob(jobId);
    
    // Update job status
    await jobService.updateJobStatus(jobId, 'cancelled');
    
    res.json({ message: 'Job cancelled' });
  } catch (error) {
    console.error('Error cancelling job:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * GET /api/jobs/:id/results
 * Get job results including molecules
 */
router.get('/:id/results', requireAuth, async (req, res) => {
  try {
    const jobId = req.params.id;
    const userId = req.user!.id;
    
    const result = await jobService.getJobWithMolecules(jobId);
    
    if (!result) {
      return res.status(404).json({ error: 'Job not found' });
    }
    
    if (result.job.userId !== userId) {
      return res.status(403).json({ error: 'Forbidden' });
    }
    
    res.json(result);
  } catch (error) {
    console.error('Error getting job results:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

export default router;
```

**Acceptance Criteria:**
- [ ] All endpoints implemented
- [ ] Authentication middleware applied
- [ ] Request validation with Zod
- [ ] Proper error handling
- [ ] Authorization checks (user owns job)

---

### Task 4.2: Create File Upload API Routes

**Objective:** Handle file uploads for job inputs.

**Steps:**
1. Install multer for file uploads:
```bash
pnpm add multer
pnpm add -D @types/multer
```

2. Create `/home/ubuntu/forcelab-elixir/server/routes/files.ts`:

```typescript
import { Router } from 'express';
import multer from 'multer';
import { fileService } from '../services/file-service';
import { jobService } from '../services/job-service';
import { requireAuth } from '../middleware/auth';

const router = Router();

// Configure multer for memory storage
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 25 * 1024 * 1024, // 25MB
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['pdb', 'cif', 'sdf', 'xyz'];
    const isValid = fileService.validateFileType(file.originalname, allowedTypes);
    
    if (isValid) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type'));
    }
  },
});

/**
 * POST /api/files/upload
 * Upload input file for a job
 */
router.post('/upload', requireAuth, upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file provided' });
    }
    
    const jobId = req.body.jobId;
    const fileType = req.body.fileType; // 'protein', 'ligand', 'pharmacophore'
    
    if (!jobId) {
      return res.status(400).json({ error: 'Job ID required' });
    }
    
    // Verify job exists and user owns it
    const job = await jobService.getJob(jobId);
    if (!job || job.userId !== req.user!.id) {
      return res.status(404).json({ error: 'Job not found' });
    }
    
    // Upload file to S3
    const result = await fileService.uploadFile(
      jobId,
      req.file.buffer,
      req.file.originalname,
      'input',
      req.file.mimetype
    );
    
    // Update job with input file reference
    const inputFiles = job.inputFiles as Record<string, string> || {};
    inputFiles[fileType] = result.key;
    
    await jobService.updateJobStatus(jobId, job.status, {
      inputFiles,
    });
    
    res.json({
      fileKey: result.key,
      fileUrl: result.url,
      fileType,
    });
  } catch (error) {
    console.error('Error uploading file:', error);
    res.status(500).json({ error: 'File upload failed' });
  }
});

/**
 * GET /api/files/:key
 * Download file
 */
router.get('/:key(*)', requireAuth, async (req, res) => {
  try {
    const fileKey = req.params.key;
    
    // Get presigned URL
    const result = await fileService.downloadFile(fileKey);
    
    // Redirect to presigned URL
    res.redirect(result.url);
  } catch (error) {
    console.error('Error downloading file:', error);
    res.status(500).json({ error: 'File download failed' });
  }
});

export default router;
```

**Acceptance Criteria:**
- [ ] File upload endpoint works
- [ ] File type validation implemented
- [ ] File size validation implemented
- [ ] Files stored in S3
- [ ] Download endpoint returns presigned URLs

---

## Phase 5: Worker Service

### Task 5.1: Create Worker Directory Structure

**Objective:** Set up the Python worker service structure.

**Steps:**
1. Create worker directory:
```bash
mkdir -p /home/ubuntu/forcelab-elixir/worker
```

2. Create directory structure:
```
worker/
├── Dockerfile
├── requirements.txt
├── worker.py
├── omtra_runner.py
└── utils.py
```

**Acceptance Criteria:**
- [ ] Worker directory created
- [ ] Directory structure in place

---

### Task 5.2: Create Worker Dockerfile

**Objective:** Create Docker image with OMTRA installed.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/worker/Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy OMTRA repository
COPY /home/ubuntu/OMTRA /srv/OMTRA

# Install CUDA dependencies
WORKDIR /srv/OMTRA
RUN uv pip install -r requirements-cuda.txt --system

# Install OMTRA
RUN uv pip install -e . --system

# Install worker dependencies
COPY requirements.txt /srv/worker/requirements.txt
WORKDIR /srv/worker
RUN pip install -r requirements.txt

# Copy worker code
COPY . /srv/worker

# Create directories
RUN mkdir -p /srv/checkpoints /tmp/omtra_jobs

# Set environment variables
ENV CHECKPOINT_DIR=/srv/checkpoints
ENV PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]
```

**Acceptance Criteria:**
- [ ] Dockerfile created
- [ ] OMTRA installation included
- [ ] Worker dependencies installed

---

### Task 5.3: Create Worker Requirements

**Objective:** Define Python dependencies for worker.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/worker/requirements.txt`:

```
redis==5.0.1
rq==1.15.1
boto3==1.34.0
requests==2.31.0
python-dotenv==1.0.0
```

**Acceptance Criteria:**
- [ ] Requirements file created
- [ ] All necessary dependencies listed

---

### Task 5.4: Create OMTRA Runner

**Objective:** Implement OMTRA CLI wrapper.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/worker/omtra_runner.py`:

```python
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class OMTRARunner:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
    
    def run_task(
        self,
        task: str,
        output_dir: Path,
        input_files: Optional[Dict[str, Path]] = None,
        parameters: Optional[Dict] = None
    ) -> Dict:
        """
        Run OMTRA task.
        
        Args:
            task: OMTRA task name
            output_dir: Directory for outputs
            input_files: Dictionary of input file paths (protein, ligand, pharmacophore)
            parameters: Additional parameters
            
        Returns:
            Dictionary with stdout, stderr, return_code
        """
        cmd = self._build_command(task, output_dir, input_files, parameters)
        
        logger.info(f"Running OMTRA command: {' '.join(map(str, cmd))}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout
                check=False
            )
            
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'success': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            logger.error("OMTRA command timed out")
            return {
                'stdout': '',
                'stderr': 'Command timed out after 600 seconds',
                'return_code': -1,
                'success': False
            }
        except Exception as e:
            logger.error(f"Error running OMTRA: {e}")
            return {
                'stdout': '',
                'stderr': str(e),
                'return_code': -1,
                'success': False
            }
    
    def _build_command(
        self,
        task: str,
        output_dir: Path,
        input_files: Optional[Dict[str, Path]],
        parameters: Optional[Dict]
    ) -> List[str]:
        """Build OMTRA CLI command."""
        cmd = ['omtra', '--task', task]
        
        # Add input files
        if input_files:
            if 'protein' in input_files:
                cmd.extend(['--protein_file', str(input_files['protein'])])
            if 'ligand' in input_files:
                cmd.extend(['--ligand_file', str(input_files['ligand'])])
            if 'pharmacophore' in input_files:
                cmd.extend(['--pharmacophore_file', str(input_files['pharmacophore'])])
        
        # Add parameters
        if parameters:
            if 'n_samples' in parameters:
                cmd.extend(['--n_samples', str(parameters['n_samples'])])
            if 'n_timesteps' in parameters:
                cmd.extend(['--n_timesteps', str(parameters['n_timesteps'])])
            if parameters.get('stochastic_sampling'):
                cmd.append('--stochastic_sampling')
            if parameters.get('compute_metrics'):
                cmd.append('--metrics')
            if parameters.get('use_gt_n_lig_atoms'):
                cmd.append('--use_gt_n_lig_atoms')
            if 'n_lig_atom_margin' in parameters:
                cmd.extend(['--n_lig_atom_margin', str(parameters['n_lig_atom_margin'])])
        
        # Add output directory
        cmd.extend(['--output_dir', str(output_dir)])
        
        return cmd
```

**Acceptance Criteria:**
- [ ] OMTRA runner class created
- [ ] Command builder implemented
- [ ] Error handling added
- [ ] Timeout handling implemented

---

### Task 5.5: Create Worker Main Script

**Objective:** Implement main worker logic with Redis queue.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/worker/worker.py`:

```python
import os
import sys
import logging
from pathlib import Path
from redis import Redis
from rq import Worker, Queue, Connection
import boto3
import requests
from omtra_runner import OMTRARunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
CHECKPOINT_DIR = Path(os.getenv('CHECKPOINT_DIR', '/srv/checkpoints'))
API_URL = os.getenv('API_URL', 'http://localhost:3000')
S3_ENDPOINT = os.getenv('S3_ENDPOINT')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_BUCKET = os.getenv('S3_BUCKET')

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

# Initialize OMTRA runner
omtra_runner = OMTRARunner(CHECKPOINT_DIR)

def download_file_from_s3(s3_key: str, local_path: Path):
    """Download file from S3 to local path."""
    logger.info(f"Downloading {s3_key} to {local_path}")
    s3_client.download_file(S3_BUCKET, s3_key, str(local_path))

def upload_file_to_s3(local_path: Path, s3_key: str):
    """Upload file from local path to S3."""
    logger.info(f"Uploading {local_path} to {s3_key}")
    s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)

def update_job_status(job_id: str, status: str, data: dict = None):
    """Update job status via API."""
    try:
        url = f"{API_URL}/api/jobs/{job_id}/status"
        payload = {'status': status}
        if data:
            payload.update(data)
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logger.info(f"Updated job {job_id} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")

def process_omtra_job(job_id: str, task: str, parameters: dict, input_files: dict = None):
    """
    Process an OMTRA job.
    
    Args:
        job_id: Unique job identifier
        task: OMTRA task name
        parameters: Task parameters
        input_files: Dictionary of input file S3 keys
    """
    logger.info(f"Processing job {job_id} with task {task}")
    
    try:
        # Update status to running
        update_job_status(job_id, 'running')
        
        # Create job directory
        job_dir = Path(f"/tmp/omtra_jobs/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)
        
        input_dir = job_dir / "inputs"
        output_dir = job_dir / "outputs"
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        # Download input files from S3
        local_input_files = {}
        if input_files:
            for file_type, s3_key in input_files.items():
                local_path = input_dir / Path(s3_key).name
                download_file_from_s3(s3_key, local_path)
                local_input_files[file_type] = local_path
        
        # Run OMTRA
        result = omtra_runner.run_task(
            task=task,
            output_dir=output_dir,
            input_files=local_input_files if local_input_files else None,
            parameters=parameters
        )
        
        if not result['success']:
            logger.error(f"OMTRA failed: {result['stderr']}")
            update_job_status(job_id, 'failed', {
                'error_message': result['stderr']
            })
            return
        
        # Upload output files to S3
        output_files = []
        for output_file in output_dir.glob('*'):
            if output_file.is_file():
                s3_key = f"jobs/{job_id}/outputs/{output_file.name}"
                upload_file_to_s3(output_file, s3_key)
                output_files.append({
                    'name': output_file.name,
                    'key': s3_key,
                    'size': output_file.stat().st_size
                })
        
        # Parse metrics if available
        metrics = None
        metrics_file = output_dir / 'metrics.json'
        if metrics_file.exists():
            import json
            with open(metrics_file) as f:
                metrics = json.load(f)
        
        # Update job status to completed
        update_job_status(job_id, 'completed', {
            'output_files': output_files,
            'metrics': metrics
        })
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
        update_job_status(job_id, 'failed', {
            'error_message': str(e)
        })

if __name__ == '__main__':
    # Connect to Redis
    redis_conn = Redis.from_url(REDIS_URL)
    
    # Create queue
    queue = Queue('omtra-jobs', connection=redis_conn)
    
    # Register job processor
    queue.register('process-omtra', process_omtra_job)
    
    # Start worker
    logger.info("Starting OMTRA worker...")
    with Connection(redis_conn):
        worker = Worker([queue])
        worker.work()
```

**Acceptance Criteria:**
- [ ] Worker script created
- [ ] Redis connection established
- [ ] Job processing logic implemented
- [ ] S3 file operations working
- [ ] Error handling and logging added

---

## Phase 6: Frontend Development

### Task 6.1: Install 3Dmol.js

**Objective:** Add molecular visualization library.

**Steps:**
1. Install 3Dmol.js:
```bash
cd /home/ubuntu/forcelab-elixir
pnpm add 3dmol
pnpm add -D @types/3dmol
```

**Acceptance Criteria:**
- [ ] Package installed
- [ ] Types available

---

### Task 6.2: Create Job Submission Form

**Objective:** Build the main job submission interface.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/src/components/jobs/JobSubmissionForm.tsx`:

```typescript
'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

const OMTRA_TASKS = [
  { value: 'denovo_ligand_condensed', label: 'De Novo Molecule Generation', description: 'Generate novel drug-like molecules from scratch' },
  { value: 'fixed_protein_ligand_denovo_condensed', label: 'Protein-Conditioned Design', description: 'Design ligands for a specific protein binding site' },
  { value: 'rigid_docking_condensed', label: 'Rigid Docking', description: 'Dock a ligand into a protein structure' },
  { value: 'ligand_conformer_condensed', label: 'Conformer Generation', description: 'Generate 3D conformations for a ligand' },
  { value: 'denovo_ligand_from_pharmacophore_condensed', label: 'Pharmacophore-Guided Design', description: 'Design molecules matching pharmacophore constraints' },
];

export function JobSubmissionForm() {
  const router = useRouter();
  const [task, setTask] = useState('');
  const [nSamples, setNSamples] = useState(100);
  const [nTimesteps, setNTimesteps] = useState(250);
  const [computeMetrics, setComputeMetrics] = useState(true);
  const [proteinFile, setProteinFile] = useState<File | null>(null);
  const [ligandFile, setLigandFile] = useState<File | null>(null);
  const [pharmacophoreFile, setPharmacophoreFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const requiresProtein = task.includes('protein');
  const requiresLigand = task.includes('docking') || task.includes('conformer');
  const requiresPharmacophore = task.includes('pharmacophore');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);

    try {
      // Create job
      const response = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task,
          parameters: {
            n_samples: nSamples,
            n_timesteps: nTimesteps,
            compute_metrics: computeMetrics,
          },
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to create job');
      }

      const job = await response.json();

      // Upload files if provided
      if (proteinFile) {
        await uploadFile(job.id, proteinFile, 'protein');
      }
      if (ligandFile) {
        await uploadFile(job.id, ligandFile, 'ligand');
      }
      if (pharmacophoreFile) {
        await uploadFile(job.id, pharmacophoreFile, 'pharmacophore');
      }

      // Redirect to job details
      router.push(`/jobs/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsSubmitting(false);
    }
  };

  const uploadFile = async (jobId: string, file: File, fileType: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('jobId', jobId);
    formData.append('fileType', fileType);

    const response = await fetch('/api/files/upload', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to upload ${fileType} file`);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Task Selection */}
      <div>
        <label className="block text-sm font-medium mb-2">Task</label>
        <select
          value={task}
          onChange={(e) => setTask(e.target.value)}
          required
          className="w-full px-4 py-2 border rounded-lg"
        >
          <option value="">Select a task...</option>
          {OMTRA_TASKS.map((t) => (
            <option key={t.value} value={t.value}>
              {t.label}
            </option>
          ))}
        </select>
        {task && (
          <p className="mt-2 text-sm text-gray-600">
            {OMTRA_TASKS.find((t) => t.value === task)?.description}
          </p>
        )}
      </div>

      {/* File Uploads */}
      {requiresProtein && (
        <div>
          <label className="block text-sm font-medium mb-2">Protein File (PDB/CIF)</label>
          <input
            type="file"
            accept=".pdb,.cif"
            onChange={(e) => setProteinFile(e.target.files?.[0] || null)}
            required
            className="w-full"
          />
        </div>
      )}

      {requiresLigand && (
        <div>
          <label className="block text-sm font-medium mb-2">Ligand File (SDF)</label>
          <input
            type="file"
            accept=".sdf"
            onChange={(e) => setLigandFile(e.target.files?.[0] || null)}
            required
            className="w-full"
          />
        </div>
      )}

      {requiresPharmacophore && (
        <div>
          <label className="block text-sm font-medium mb-2">Pharmacophore File (XYZ)</label>
          <input
            type="file"
            accept=".xyz"
            onChange={(e) => setPharmacophoreFile(e.target.files?.[0] || null)}
            className="w-full"
          />
        </div>
      )}

      {/* Parameters */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-2">Number of Samples</label>
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
          <label className="block text-sm font-medium mb-2">Number of Timesteps</label>
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

      {/* Options */}
      <div>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={computeMetrics}
            onChange={(e) => setComputeMetrics(e.target.checked)}
            className="mr-2"
          />
          <span className="text-sm">Compute metrics</span>
        </label>
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-600">
          {error}
        </div>
      )}

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isSubmitting || !task}
        className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isSubmitting ? 'Submitting...' : 'Submit Job'}
      </button>
    </form>
  );
}
```

**Acceptance Criteria:**
- [ ] Form component created
- [ ] Task selection dropdown
- [ ] Conditional file upload fields
- [ ] Parameter inputs
- [ ] Form validation
- [ ] Submit handler with API integration

---

### Task 6.3: Create Molecular Viewer Component

**Objective:** Implement 3D molecular visualization.

**Steps:**
1. Create `/home/ubuntu/forcelab-elixir/src/components/molecules/MolecularViewer.tsx`:

```typescript
'use client';

import { useEffect, useRef } from 'react';
import * as $3Dmol from '3dmol';

interface MolecularViewerProps {
  moleculeData: string; // SDF format
  proteinData?: string; // PDB format
  pharmacophoreData?: string; // XYZ format
  style?: 'stick' | 'cartoon' | 'surface' | 'sphere';
}

export function MolecularViewer({
  moleculeData,
  proteinData,
  pharmacophoreData,
  style = 'stick',
}: MolecularViewerProps) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const viewerInstanceRef = useRef<any>(null);

  useEffect(() => {
    if (!viewerRef.current) return;

    // Initialize viewer
    const viewer = $3Dmol.createViewer(viewerRef.current, {
      backgroundColor: 'white',
    });
    viewerInstanceRef.current = viewer;

    // Add protein if provided
    if (proteinData) {
      viewer.addModel(proteinData, 'pdb');
      viewer.setStyle({}, { cartoon: { color: 'spectrum' } });
    }

    // Add molecule
    if (moleculeData) {
      const model = viewer.addModel(moleculeData, 'sdf');
      
      if (style === 'stick') {
        viewer.setStyle({ model }, { stick: { colorscheme: 'default' } });
      } else if (style === 'sphere') {
        viewer.setStyle({ model }, { sphere: { colorscheme: 'default' } });
      } else if (style === 'surface') {
        viewer.addSurface($3Dmol.SurfaceType.VDW, { opacity: 0.7 }, { model });
      }
    }

    // Add pharmacophore if provided
    if (pharmacophoreData) {
      // Parse XYZ and add spheres for pharmacophore features
      const lines = pharmacophoreData.split('\n');
      for (let i = 2; i < lines.length; i++) {
        const parts = lines[i].trim().split(/\s+/);
        if (parts.length >= 4) {
          const [type, x, y, z] = parts;
          viewer.addSphere({
            center: { x: parseFloat(x), y: parseFloat(y), z: parseFloat(z) },
            radius: 1.0,
            color: getPharmacophoreColor(type),
            alpha: 0.5,
          });
        }
      }
    }

    viewer.zoomTo();
    viewer.render();

    return () => {
      viewer.clear();
    };
  }, [moleculeData, proteinData, pharmacophoreData, style]);

  const getPharmacophoreColor = (type: string): string => {
    const colors: Record<string, string> = {
      'Hydrophobic': 'green',
      'Aromatic': 'orange',
      'HBond Acceptor': 'red',
      'HBond Donor': 'blue',
      'Positive': 'cyan',
      'Negative': 'magenta',
    };
    return colors[type] || 'gray';
  };

  return (
    <div
      ref={viewerRef}
      className="w-full h-[600px] border rounded-lg"
      style={{ position: 'relative' }}
    />
  );
}
```

**Acceptance Criteria:**
- [ ] Viewer component created
- [ ] 3Dmol.js integration working
- [ ] Protein rendering
- [ ] Ligand rendering
- [ ] Pharmacophore visualization
- [ ] Interactive controls (rotate, zoom)

---

## Phase 7: Testing & Deployment

### Task 7.1: Write Unit Tests

**Objective:** Test core functionality.

**Steps:**
1. Create test files for services
2. Run tests:
```bash
pnpm test
```

**Acceptance Criteria:**
- [ ] Job service tests pass
- [ ] File service tests pass
- [ ] Queue service tests pass

---

### Task 7.2: Integration Testing

**Objective:** Test end-to-end workflows.

**Steps:**
1. Test job submission with sample files
2. Verify worker processes jobs
3. Check results are displayed correctly

**Acceptance Criteria:**
- [ ] Complete job workflow works
- [ ] All OMTRA tasks tested
- [ ] Error handling verified

---

### Task 7.3: Save Checkpoint and Deploy

**Objective:** Deploy to production.

**Steps:**
1. Mark all completed items in todo.md as [x]
2. Save checkpoint using Manus webdev tools
3. Click Publish button in Manus UI

**Acceptance Criteria:**
- [ ] All todo items marked complete
- [ ] Checkpoint saved
- [ ] Application deployed
- [ ] Production testing completed

---

## Summary

This checklist provides a complete, step-by-step guide for implementing OMTRA integration into ForcelabElixir. Each task includes:

- Clear objectives
- Detailed implementation steps
- Code examples
- Acceptance criteria

Follow the tasks in order, marking each as complete in todo.md as you progress. The implementation should take approximately 20-30 hours of development time for an experienced full-stack developer familiar with the technologies involved.

**Key Integration Points:**

1. **Database:** PostgreSQL with Drizzle ORM for job and molecule storage
2. **Queue:** Redis + Bull for asynchronous job processing
3. **Storage:** S3 for file uploads and results
4. **Worker:** Python service running OMTRA CLI
5. **Frontend:** React/Next.js with 3Dmol.js for visualization
6. **API:** RESTful endpoints for job management

**Critical Success Factors:**

- OMTRA model checkpoints must be downloaded and accessible to worker
- GPU access required for worker container
- Proper error handling throughout the stack
- Real-time job status updates for good UX
- Comprehensive testing of all OMTRA task types

---

**End of Checklist**
