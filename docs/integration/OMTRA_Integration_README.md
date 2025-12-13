# OMTRA-ForcelabElixir Integration Documentation Package

**Author:** Manus AI  
**Date:** December 9, 2025  
**Version:** 1.0

---

## Overview

This documentation package provides everything needed to integrate **OMTRA** (a multi-task generative model for structure-based drug design) into the **ForcelabElixir** stack. The integration creates a unified web platform combining OMTRA's AI-driven molecular generation capabilities with a modern full-stack architecture.

---

## What is OMTRA?

OMTRA is a flow-matching based generative model for small-molecule and protein systems developed by the Koes Lab at the University of Pittsburgh. It supports various tasks relevant to structure-based drug design, including:

- **Unconditional 3D de novo molecule generation**
- **Protein pocket-conditioned molecule design**
- **Protein-ligand docking** (rigid and flexible)
- **Conformer generation**
- **Pharmacophore-conditioned molecule generation**
- **Combined protein and pharmacophore-conditioned design**

OMTRA is described in their preprint available at [https://arxiv.org/abs/2512.05080](https://arxiv.org/abs/2512.05080) and will be presented at MLSB 2025.

---

## What is ForcelabElixir?

ForcelabElixir is an integrated platform for structure-based drug design that combines cutting-edge AI models with user-friendly web interfaces. The platform provides researchers and drug designers with powerful tools for molecular generation, docking, and analysis through an accessible web application.

---

## Documentation Package Contents

This package contains three comprehensive documents designed to guide the implementation process:

### 1. Technical Specification (`OMTRA_ForcelabElixir_Integration_Spec.md`)

**Purpose:** Comprehensive technical specification and architecture design

**Contents:**
- Executive summary and project overview
- OMTRA capabilities and task descriptions
- High-level architecture design with diagrams
- Complete technology stack specification
- Detailed system components breakdown
- Database schema definitions
- API endpoint specifications
- Frontend component specifications
- Integration steps and phases
- Configuration requirements
- Deployment strategy
- Testing requirements
- Appendices with detailed task descriptions and file formats

**Use this document for:**
- Understanding the overall architecture
- Making design decisions
- Reference during implementation
- Onboarding new team members
- System documentation

**Length:** ~40 pages  
**Detail Level:** High - includes rationale and design decisions

---

### 2. Implementation Checklist (`OMTRA_Implementation_Checklist.md`)

**Purpose:** Step-by-step implementation guide with detailed tasks

**Contents:**
- Phased implementation approach (7 phases)
- Detailed task breakdown with acceptance criteria
- Complete code examples for each component
- Database schema with Drizzle ORM
- Backend services implementation
- API routes with authentication
- Worker service setup
- Frontend components with React/Next.js
- Testing requirements
- Deployment steps

**Use this document for:**
- Following a structured implementation path
- Tracking progress (mark tasks as complete)
- Code templates and examples
- Ensuring nothing is missed
- Quality assurance checkpoints

**Length:** ~50 pages  
**Detail Level:** Very High - includes complete code examples

---

### 3. Quick Start Guide (`OMTRA_Quick_Start_Guide.md`)

**Purpose:** Rapid implementation reference for experienced developers

**Contents:**
- Essential setup commands
- Database schema templates
- Backend service templates
- API route templates
- Worker service templates
- Frontend component templates
- Environment variables
- Docker Compose configuration
- Testing commands
- Common issues and solutions
- Implementation priority order

**Use this document for:**
- Quick reference during coding
- Copy-paste code templates
- Troubleshooting common issues
- Setting up development environment
- ChatGPT Codex integration

**Length:** ~15 pages  
**Detail Level:** Medium - focused on code and commands

---

## Implementation Approach

### Recommended Workflow

**For Complete Implementation:**

1. **Read** the Technical Specification document first to understand the architecture
2. **Follow** the Implementation Checklist sequentially, marking tasks complete
3. **Reference** the Quick Start Guide for code templates and commands
4. **Test** each component as you build it
5. **Document** any deviations or custom implementations
6. **Deploy** using the provided deployment strategy

**For ChatGPT Codex:**

1. **Upload** all three documents to ChatGPT Codex
2. **Start** with the Quick Start Guide for immediate context
3. **Reference** the Implementation Checklist for task order
4. **Consult** the Technical Specification for design decisions
5. **Iterate** on each phase, testing as you go

---

## Key Technologies

### Frontend Stack
- **React 18** with **Next.js 14** (App Router)
- **Tailwind CSS 4** for styling
- **3Dmol.js** for molecular visualization
- **TypeScript** for type safety
- **SWR** for data fetching

### Backend Stack
- **Node.js 22** with **TypeScript**
- **Express.js** or **Fastify** for API
- **Drizzle ORM** with **PostgreSQL**
- **JWT** authentication with OAuth2
- **Bull** with **Redis** for job queue

### Worker Stack
- **Python 3.11**
- **OMTRA** (PyTorch, DGL, RDKit)
- **RQ** (Redis Queue) for job processing
- **Boto3** for S3 integration

### Infrastructure
- **PostgreSQL 15+** for database
- **Redis 7** for queue and caching
- **S3-compatible storage** for files
- **Docker** for containerization
- **Manus hosting** for deployment

---

## System Architecture

The ForcelabElixir platform follows a microservices-inspired architecture:

```
┌──────────────────────────────────────────────────┐
│         Frontend (React/Next.js)                 │
│  - Job Submission Interface                      │
│  - 3D Molecular Viewer                          │
│  - Job Status Dashboard                          │
└──────────────────────────────────────────────────┘
                    ↓ HTTP/REST
┌──────────────────────────────────────────────────┐
│         Backend API (Node.js)                    │
│  - Authentication & Authorization                │
│  - Job Management                                │
│  - File Upload/Download                          │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│         Job Queue (Redis + Bull)                 │
│  - Task Queueing                                │
│  - Job Status Tracking                           │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│         Worker Service (Python)                  │
│  - OMTRA CLI Integration                        │
│  - Molecular Generation                          │
│  - Result Processing                             │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│         Database (PostgreSQL)                    │
│  - User Data, Job Records                       │
│  - Generated Molecules, Metrics                  │
└──────────────────────────────────────────────────┘
```

---

## Prerequisites

### System Requirements
- **Linux System** (Ubuntu 22.04 recommended)
- **NVIDIA GPU** with CUDA support (CUDA 12.1 recommended)
- **Python 3.11**
- **Node.js 22**
- **PostgreSQL 15+**
- **Redis 7**
- **Docker** and **Docker Compose**

### Knowledge Requirements
- **Full-stack web development** (React, Node.js)
- **Database design** and SQL
- **REST API** design and implementation
- **Python** programming
- **Docker** and containerization
- **Basic understanding** of molecular modeling (helpful but not required)

### Time Estimate
- **Experienced Developer:** 20-30 hours
- **Intermediate Developer:** 40-50 hours
- **With ChatGPT Codex:** 10-15 hours (with review)

---

## Quick Start

### 1. Clone OMTRA Repository

```bash
cd /home/ubuntu
gh repo clone Mino01/OMTRA
```

### 2. Download Model Checkpoints

```bash
cd /home/ubuntu/OMTRA
mkdir -p omtra/trained_models
wget -r -np -nH --cut-dirs=3 -R "index.html*" \
  -P omtra/trained_models \
  https://bits.csb.pitt.edu/files/OMTRA/omtra_v0_weights/
```

### 3. Initialize ForcelabElixir Project

Use Manus webdev tools to initialize the project with `web-db-user` features.

### 4. Follow Implementation Checklist

Work through each phase in the Implementation Checklist document, marking tasks complete as you go.

### 5. Test and Deploy

Test the complete workflow, save a checkpoint, and deploy using Manus hosting.

---

## Key Features

### User Features
- **Job Submission:** Submit molecular generation jobs with custom parameters
- **File Upload:** Upload protein, ligand, and pharmacophore files
- **3D Visualization:** Interactive molecular viewer with 3Dmol.js
- **Job Tracking:** Real-time job status updates and progress monitoring
- **Results Download:** Download generated molecules and metrics
- **User Authentication:** Secure login with OAuth2 support
- **Job History:** View and manage all submitted jobs

### Technical Features
- **Asynchronous Processing:** Jobs processed in background workers
- **Scalable Architecture:** Horizontal scaling of worker services
- **File Storage:** S3-compatible object storage for inputs and outputs
- **Database Persistence:** PostgreSQL for reliable data storage
- **API-First Design:** RESTful API for all operations
- **Type Safety:** TypeScript throughout the stack
- **Containerization:** Docker for consistent deployment
- **Queue Management:** Redis-based job queue with retry logic

---

## Database Schema

### Core Tables

**users** - User accounts and authentication  
**jobs** - Job records with status and parameters  
**molecules** - Generated molecules with properties  
**job_files** - Uploaded and generated files

### Relationships

- Users have many Jobs (one-to-many)
- Jobs have many Molecules (one-to-many)
- Jobs have many Job Files (one-to-many)

### Indexes

Optimized indexes on:
- User ID for job queries
- Job status for filtering
- Created timestamps for sorting
- SMILES strings for molecule search

---

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/me` - Get current user

### Jobs
- `POST /api/jobs` - Create new job
- `GET /api/jobs` - List user's jobs
- `GET /api/jobs/:id` - Get job details
- `DELETE /api/jobs/:id` - Delete job
- `POST /api/jobs/:id/cancel` - Cancel job

### Files
- `POST /api/files/upload` - Upload file
- `GET /api/files/:key` - Download file

### Results
- `GET /api/jobs/:id/results` - Get job results
- `GET /api/jobs/:id/molecules` - List molecules
- `GET /api/jobs/:id/metrics` - Get metrics
- `GET /api/jobs/:id/download` - Download all results

---

## OMTRA Tasks Supported

### Unconditional Generation
- `denovo_ligand_condensed` - Generate novel molecules

### Protein-Conditioned
- `fixed_protein_ligand_denovo_condensed` - Design for fixed protein
- `protein_ligand_denovo_condensed` - Design with flexible protein

### Docking
- `rigid_docking_condensed` - Rigid docking
- `flexible_docking_condensed` - Flexible docking
- `expapo_conditioned_ligand_docking_condensed` - Docking from apo
- `predapo_conditioned_ligand_docking_condensed` - Docking from predicted apo

### Conformer Generation
- `ligand_conformer_condensed` - Generate conformers

### Pharmacophore-Conditioned
- `denovo_ligand_pharmacophore_condensed` - Joint generation
- `denovo_ligand_from_pharmacophore_condensed` - From pharmacophore
- `ligand_conformer_from_pharmacophore_condensed` - Conformer from pharmacophore
- `fixed_protein_pharmacophore_ligand_denovo_condensed` - Protein + pharmacophore
- `rigid_docking_pharmacophore_condensed` - Docking with pharmacophore

---

## Testing Strategy

### Unit Tests
- Job service CRUD operations
- File upload/download
- Authentication middleware
- Queue service operations
- Frontend component rendering

### Integration Tests
- Complete job workflow
- File upload and processing
- Worker job processing
- API endpoint integration

### End-to-End Tests
- User creates account
- User submits job
- Job is processed
- Results are displayed
- User downloads results

---

## Deployment

### Local Development
1. Start PostgreSQL and Redis
2. Run database migrations
3. Start development server
4. Start worker service

### Production Deployment
1. Build Docker images
2. Deploy to Manus hosting
3. Configure environment variables
4. Verify model checkpoints
5. Test job submission
6. Monitor logs

---

## Troubleshooting

### Common Issues

**Issue:** Model checkpoints not found  
**Solution:** Download checkpoints to correct directory

**Issue:** Worker can't connect to Redis  
**Solution:** Verify REDIS_URL environment variable

**Issue:** GPU not available  
**Solution:** Add GPU support to Docker configuration

**Issue:** File upload fails  
**Solution:** Check S3 credentials and permissions

**Issue:** Job stays in pending status  
**Solution:** Check worker logs for errors

---

## Support and Resources

### OMTRA Resources
- **GitHub:** https://github.com/gnina/OMTRA
- **Paper:** https://arxiv.org/abs/2512.05080
- **Documentation:** See OMTRA readme.md

### Manus Resources
- **Documentation:** https://docs.manus.im
- **Support:** https://help.manus.im
- **Web Development:** https://docs.manus.im/webdev

### Related Technologies
- **3Dmol.js:** https://3dmol.csb.pitt.edu/
- **RDKit:** https://www.rdkit.org/
- **Drizzle ORM:** https://orm.drizzle.team/
- **Next.js:** https://nextjs.org/docs
- **Bull Queue:** https://github.com/OptimalBits/bull

---

## Contributing

When extending this integration:

1. **Follow the established architecture** - Maintain separation of concerns
2. **Add tests** for new features
3. **Update documentation** when making changes
4. **Use TypeScript** for type safety
5. **Follow code style** conventions
6. **Add error handling** for all operations
7. **Log important events** for debugging

---

## Future Enhancements

Potential improvements and extensions:

### Short-term
- Add batch job submission
- Implement job scheduling
- Add email notifications
- Create admin dashboard
- Add usage analytics

### Medium-term
- Support for custom OMTRA models
- Integration with molecular databases
- Advanced filtering and search
- Collaborative features (share jobs)
- API rate limiting and quotas

### Long-term
- Multi-model support (other generative models)
- Workflow automation
- Integration with experimental data
- Machine learning on generated molecules
- Cloud-native deployment options

---

## License

This integration documentation is provided as-is for use with the OMTRA project and ForcelabElixir platform. OMTRA itself is licensed under Apache 2.0. Please refer to the original OMTRA repository for licensing details.

---

## Acknowledgments

- **OMTRA Development Team** at the Koes Lab, University of Pittsburgh
- **Manus Platform** for hosting and development tools
- **Open Source Community** for the underlying technologies

---

## Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | December 9, 2025 | Initial release |

---

## Contact

For questions about this integration:
- Review the documentation in this package
- Consult the OMTRA GitHub repository
- Submit issues to the ForcelabElixir project

For questions about Manus platform:
- Visit https://help.manus.im

---

**End of README**

---

## Next Steps

1. **Read** the Technical Specification document for architecture understanding
2. **Follow** the Implementation Checklist for step-by-step guidance
3. **Use** the Quick Start Guide for code templates
4. **Test** thoroughly at each phase
5. **Deploy** to production when ready

Good luck with your implementation!
