import { SemanticEngine } from '../engines/semantic-engine.js';
import { PatternEngine } from '../engines/pattern-engine.js';
import { SurrealStorage } from '../storage/surreal-storage.js';
import { config } from '../config/config.js';
import { nanoid } from 'nanoid';

export interface LearningResult {
  success: boolean;
  conceptsLearned: number;
  patternsLearned: number;
  featuresLearned: number;
  insights: string[];
  timeElapsed: number;
  blueprint?: {
    techStack: string[];
    entryPoints: Record<string, string>;
    keyDirectories: Record<string, string>;
    architecture: string;
  };
}

export interface LearningOptions {
  force?: boolean;
  progressCallback?: (current: number, total: number, message: string) => void;
}

/**
 * Shared learning service used by both CLI and MCP tools
 * Ensures consistent behavior across all interfaces
 */
export class LearningService {
  /**
   * Learn from a codebase - shared implementation for CLI and MCP
   */
  static async learnFromCodebase(
    path: string,
    options: LearningOptions = {}
  ): Promise<LearningResult> {
    const startTime = Date.now();
    const insights: string[] = [];

    // Create project-specific storage and engines
    const projectDbPath = config.getDatabasePath(path);
    const projectStorage = await SurrealStorage.create(projectDbPath);
    const projectSemanticEngine = new SemanticEngine(projectStorage);
    const projectPatternEngine = new PatternEngine(projectStorage);

    try {
      console.error(`🗄️ Using project database: ${projectDbPath}`);

      // Check if already learned and not forcing re-learn
      if (!options.force) {
        const existingIntelligence = await this.checkExistingIntelligence(projectStorage, path);
        if (existingIntelligence && existingIntelligence.concepts > 0) {
          return {
            success: true,
            conceptsLearned: existingIntelligence.concepts,
            patternsLearned: existingIntelligence.patterns,
            featuresLearned: existingIntelligence.features,
            insights: ['Using existing intelligence (use force: true to re-learn)'],
            timeElapsed: Date.now() - startTime
          };
        }
      }

      // Phase 1: Comprehensive codebase analysis
      insights.push('🔍 Phase 1: Analyzing codebase structure...');
      const codebaseAnalysis = await projectSemanticEngine.analyzeCodebase(path);
      insights.push(`   ✅ Detected languages: ${codebaseAnalysis.languages?.join(', ') || 'unknown'}`);
      insights.push(`   ✅ Found frameworks: ${codebaseAnalysis.frameworks?.join(', ') || 'none detected'}`);
      if (codebaseAnalysis.complexity) {
        insights.push(`   ✅ Complexity: ${codebaseAnalysis.complexity.cyclomatic?.toFixed(1) || 'N/A'} cyclomatic, ${codebaseAnalysis.complexity.cognitive?.toFixed(1) || 'N/A'} cognitive`);
      }

      // Phase 2: Deep semantic learning
      insights.push('🧠 Phase 2: Learning semantic concepts...');
      const concepts = await projectSemanticEngine.learnFromCodebase(path, options.progressCallback);

      // Analyze concept distribution
      const conceptTypes = concepts.reduce((acc, concept) => {
        const type = concept.type || 'unknown';
        acc[type] = (acc[type] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

      insights.push(`   ✅ Extracted ${concepts.length} semantic concepts:`);
      Object.entries(conceptTypes).forEach(([type, count]) => {
        insights.push(`     - ${count} ${type}${count > 1 ? 's' : ''}`);
      });

      // Phase 3: Pattern discovery and learning
      insights.push('🔄 Phase 3: Discovering coding patterns...');
      const patterns = await projectPatternEngine.learnFromCodebase(path, options.progressCallback);

      // Analyze pattern distribution
      const patternTypes = patterns.reduce((acc, pattern) => {
        const patternType = pattern.type || 'unknown';
        const category = patternType.split('_')[0];
        acc[category] = (acc[category] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

      insights.push(`   ✅ Identified ${patterns.length} coding patterns:`);
      Object.entries(patternTypes).forEach(([category, count]) => {
        insights.push(`     - ${count} ${category} pattern${count > 1 ? 's' : ''}`);
      });

      // Phase 4: Relationship and dependency analysis
      insights.push('🔗 Phase 4: Analyzing relationships and dependencies...');
      const relationships = await this.analyzeCodebaseRelationships(concepts, patterns);
      insights.push(`   ✅ Built ${relationships.conceptRelationships} concept relationships`);
      insights.push(`   ✅ Identified ${relationships.dependencyPatterns} dependency patterns`);

      // Phase 5: Intelligence synthesis and storage
      insights.push('💾 Phase 5: Synthesizing and storing intelligence...');
      await this.storeIntelligence(projectStorage, path, concepts, patterns);

      // Generate learning insights
      const learningInsights = await this.generateLearningInsights(concepts, patterns, codebaseAnalysis);
      insights.push('🎯 Learning Summary:');
      learningInsights.forEach(insight => insights.push(`   ${insight}`));

      // Phase 6: Vector embeddings for semantic search
      insights.push('🔍 Phase 6: Building semantic search index...');
      insights.push(`   ✅ Using free local embeddings (transformers.js)`);

      const vectorCount = await this.buildSemanticIndex(projectStorage, concepts, patterns);
      insights.push(`   ✅ Created ${vectorCount} vector embeddings for semantic search`);

      // Phase 6.5: Ensure project metadata exists (required for foreign keys)
      insights.push('📋 Phase 6.5: Ensuring project metadata...');
      const existingMetadata = await projectStorage.getProjectMetadata(path);
      if (!existingMetadata) {
        await projectStorage.insertProjectMetadata({
          projectId: nanoid(),
          projectPath: path,
          projectName: path.split('/').pop() || 'unknown',
          languagePrimary: codebaseAnalysis.languages?.[0],
          languagesDetected: codebaseAnalysis.languages || [],
          frameworkDetected: codebaseAnalysis.frameworks || [],
          intelligenceVersion: '0.6.0',
          lastFullScan: new Date()
        });
        insights.push(`   ✅ Created project metadata for ${path}`);
      } else {
        insights.push(`   ✅ Project metadata already exists`);
      }

      // Phase 7: Feature mapping for intelligent file routing
      insights.push('🗺️  Phase 7: Building feature map for file routing...');
      const featureMaps = await projectPatternEngine.buildFeatureMap(path);

      // Debug: Log what we got from buildFeatureMap
      if (featureMaps.length > 0) {
        insights.push(`   🔍 Debug: Received ${featureMaps.length} feature maps from buildFeatureMap`);
        const sample = featureMaps[0];
        insights.push(`   🔍 Debug: Sample feature map: id=${sample?.id}, featureName=${sample?.featureName}, primaryFiles=${sample?.primaryFiles?.length || 0}`);
      }

      // Store feature maps in database (filter out invalid ones)
      let validFeatureMaps = 0;
      let skippedDueToMissingId = 0;
      let skippedDueToMissingName = 0;
      for (const featureMap of featureMaps) {
        // Skip feature maps with missing required fields
        if (!featureMap.id) {
          skippedDueToMissingId++;
          continue;
        }
        if (!featureMap.featureName) {
          skippedDueToMissingName++;
          continue;
        }

        await projectStorage.insertFeatureMap({
          id: featureMap.id,
          projectPath: path,
          featureName: featureMap.featureName,
          primaryFiles: featureMap.primaryFiles || [],
          relatedFiles: featureMap.relatedFiles || [],
          dependencies: featureMap.dependencies || [],
          status: 'active'
        });
        validFeatureMaps++;
      }
      const totalFiles = featureMaps.reduce((sum, fm) => sum + (fm.primaryFiles?.length || 0) + (fm.relatedFiles?.length || 0), 0);
      const totalSkipped = skippedDueToMissingId + skippedDueToMissingName;
      if (totalSkipped > 0) {
        insights.push(`   ⚠️  Skipped ${totalSkipped} invalid feature maps (${skippedDueToMissingId} missing ID, ${skippedDueToMissingName} missing name)`);
      }
      insights.push(`   ✅ Mapped ${validFeatureMaps} features to ${totalFiles} files`);

      // Phase 8: Store blueprint data
      insights.push('💾 Phase 8: Storing project blueprint...');
      await this.storeProjectBlueprint(path, codebaseAnalysis, projectStorage);

      const timeElapsed = Date.now() - startTime;
      insights.push(`⚡ Learning completed in ${timeElapsed}ms`);

      // Build blueprint summary
      const blueprint = {
        techStack: codebaseAnalysis.frameworks || [],
        entryPoints: (codebaseAnalysis.entryPoints || []).reduce((acc, ep) => {
          acc[ep.type] = ep.filePath;
          return acc;
        }, {} as Record<string, string>),
        keyDirectories: (codebaseAnalysis.keyDirectories || []).reduce((acc, dir) => {
          acc[dir.type] = dir.path;
          return acc;
        }, {} as Record<string, string>),
        architecture: this.inferArchitecturePattern(codebaseAnalysis)
      };

      return {
        success: true,
        conceptsLearned: concepts.length,
        patternsLearned: patterns.length,
        featuresLearned: validFeatureMaps,
        insights,
        timeElapsed,
        blueprint
      };
    } catch (error) {
      return {
        success: false,
        conceptsLearned: 0,
        patternsLearned: 0,
        featuresLearned: 0,
        insights: [`❌ Learning failed: ${error instanceof Error ? error.message : error}`],
        timeElapsed: Date.now() - startTime
      };
    } finally {
      // Clean up project-specific resources
      if (projectSemanticEngine) {
        projectSemanticEngine.cleanup();
      }
      if (projectStorage) {
        try {
          await projectStorage.close();
        } catch (error) {
          console.warn('Warning: Failed to close project storage:', error);
        }
      }
    }
  }

  private static async checkExistingIntelligence(
    storage: SurrealStorage,
    path: string
  ): Promise<{ concepts: number; patterns: number; features: number } | null> {
    const concepts = (await storage.getSemanticConcepts()).length;
    const patterns = (await storage.getDeveloperPatterns()).length;
    const features = (await storage.getFeatureMaps(path)).length;

    if (concepts > 0 || patterns > 0) {
      return { concepts, patterns, features };
    }

    console.warn('⚠️  No existing intelligence found in project database - starting fresh analysis');
    return null;
  }

  private static async storeIntelligence(
    storage: SurrealStorage,
    _path: string,
    concepts: any[],
    patterns: any[]
  ): Promise<void> {
    // Store concepts
    for (const concept of concepts) {
      await storage.insertSemanticConcept({
        id: concept.id,
        conceptName: concept.name,
        conceptType: concept.type,
        confidenceScore: concept.confidence,
        relationships: concept.relationships,
        evolutionHistory: {},
        filePath: concept.filePath,
        lineRange: concept.lineRange
      });
    }

    // Store patterns
    for (const pattern of patterns) {
      await storage.insertDeveloperPattern({
        patternId: pattern.id,
        patternType: pattern.type,
        patternContent: pattern.content,
        frequency: pattern.frequency,
        contexts: pattern.contexts,
        examples: pattern.examples,
        confidence: pattern.confidence
      });
    }
  }

  private static async analyzeCodebaseRelationships(
    concepts: any[],
    patterns: any[]
  ): Promise<{ conceptRelationships: number; dependencyPatterns: number }> {
    const conceptRelationships = new Set<string>();

    // Group concepts by file
    const conceptsByFile = concepts.reduce((acc, concept) => {
      const filePath = concept.filePath || concept.file_path || 'unknown';
      if (!acc[filePath]) acc[filePath] = [];
      acc[filePath].push(concept);
      return acc;
    }, {} as Record<string, any[]>);

    // Find relationships within files
    Object.values(conceptsByFile).forEach(fileConcepts => {
      if (Array.isArray(fileConcepts)) {
        for (let i = 0; i < fileConcepts.length; i++) {
          for (let j = i + 1; j < fileConcepts.length; j++) {
            const relationshipKey = `${fileConcepts[i].id}-${fileConcepts[j].id}`;
            conceptRelationships.add(relationshipKey);
          }
        }
      }
    });

    // Analyze dependency patterns
    const dependencyPatterns = new Set<string>();
    patterns.forEach(pattern => {
      const patternType = pattern.type || '';
      if (patternType.includes('dependency') ||
        patternType.includes('import') ||
        patternType.includes('organization')) {
        dependencyPatterns.add(pattern.id);
      }
    });

    return {
      conceptRelationships: conceptRelationships.size,
      dependencyPatterns: dependencyPatterns.size
    };
  }

  private static async generateLearningInsights(
    concepts: any[],
    patterns: any[],
    codebaseAnalysis: any
  ): Promise<string[]> {
    const insights: string[] = [];

    // Analyze codebase characteristics
    const totalLines = codebaseAnalysis?.complexity?.lines || 0;
    const conceptDensity = totalLines > 0 ? (concepts.length / totalLines * 1000).toFixed(2) : '0';
    if (totalLines > 0) {
      insights.push(`📊 Concept density: ${conceptDensity} concepts per 1000 lines`);
    }

    // Analyze pattern distribution
    const namingPatterns = patterns.filter(p => p.type?.includes('naming'));
    const structuralPatterns = patterns.filter(p => p.type?.includes('organization') || p.type?.includes('structure'));
    const implementationPatterns = patterns.filter(p => p.type?.includes('implementation'));

    if (namingPatterns.length > 0) {
      insights.push(`✨ Strong naming conventions detected (${namingPatterns.length} patterns)`);
    }
    if (structuralPatterns.length > 0) {
      insights.push(`🏗️ Organized code structure found (${structuralPatterns.length} patterns)`);
    }
    if (implementationPatterns.length > 0) {
      insights.push(`⚙️ Design patterns in use (${implementationPatterns.length} patterns)`);
    }

    // Analyze complexity
    const complexity = codebaseAnalysis?.complexity;
    if (complexity && typeof complexity.cyclomatic === 'number') {
      if (complexity.cyclomatic < 10) {
        insights.push('🟢 Low complexity codebase - easy to maintain');
      } else if (complexity.cyclomatic < 30) {
        insights.push('🟡 Moderate complexity - consider refactoring high-complexity areas');
      } else {
        insights.push('🔴 High complexity detected - refactoring recommended');
      }
    }

    // Analyze language and framework usage
    const languages = codebaseAnalysis?.languages || [];
    const frameworks = codebaseAnalysis?.frameworks || [];

    if (languages.length === 1) {
      insights.push(`🎯 Single-language codebase (${languages[0]}) - consistent technology stack`);
    } else if (languages.length > 1) {
      insights.push(`🌐 Multi-language codebase (${languages.join(', ')}) - consider integration patterns`);
    }

    if (frameworks.length > 0) {
      insights.push(`🔧 Framework usage: ${frameworks.join(', ')}`);
    }

    return insights;
  }

  private static async buildSemanticIndex(
    storage: SurrealStorage,
    concepts: any[],
    patterns: any[]
  ): Promise<number> {
    try {
      let vectorCount = 0;

      // Create embeddings for semantic concepts
      for (const concept of concepts) {
        const conceptType = concept.type || 'unknown';
        const text = `${concept.name} ${conceptType}`;
        await storage.storeCodeEmbedding(text, {
          id: concept.id,
          filePath: concept.filePath,
          functionName: conceptType === 'function' ? concept.name : undefined,
          className: conceptType === 'class' ? concept.name : undefined,
          language: 'unknown',
          complexity: 1,
          lineCount: 1,
          lastModified: new Date()
        });
        vectorCount++;
      }

      // Create embeddings for patterns
      for (const pattern of patterns) {
        const patternType = pattern.type || 'unknown';
        const text = `${patternType} ${pattern.content?.description || ''}`;
        await storage.storeCodeEmbedding(text, {
          id: pattern.id,
          filePath: `pattern-${patternType}`,
          language: 'pattern',
          complexity: pattern.frequency || 1,
          lineCount: 1,
          lastModified: new Date()
        });
        vectorCount++;
      }

      return vectorCount;
    } catch (error) {
      console.warn('Failed to build semantic index:', error);
      return 0;
    }
  }

  private static async storeProjectBlueprint(
    projectPath: string,
    codebaseAnalysis: any,
    storage: SurrealStorage
  ): Promise<void> {
    // Debug logging to track what's being received
    // console.error(`\n🔍 storeProjectBlueprint called for project: ${projectPath}`);
    // console.error(`   codebaseAnalysis.entryPoints: ${codebaseAnalysis.entryPoints ? `Array(${codebaseAnalysis.entryPoints.length})` : 'undefined'}`);
    // console.error(`   codebaseAnalysis.keyDirectories: ${codebaseAnalysis.keyDirectories ? `Array(${codebaseAnalysis.keyDirectories.length})` : 'undefined'}`);

    // Store entry points (filter out invalid ones)
    if (codebaseAnalysis.entryPoints && Array.isArray(codebaseAnalysis.entryPoints)) {
      // let stored = 0;
      // let skipped = 0;

      for (const entryPoint of codebaseAnalysis.entryPoints) {
        // Skip entry points with missing required fields
        if (!entryPoint.type || !entryPoint.filePath) {
          // skipped++;
          // console.error(`   ⚠️  Skipping invalid entry point: type=${entryPoint.type}, filePath=${entryPoint.filePath}`);
          continue;
        }

        await storage.insertEntryPoint({
          id: nanoid(),
          projectPath,
          entryType: entryPoint.type,
          filePath: entryPoint.filePath,
          description: entryPoint.description,
          framework: entryPoint.framework
        });
        // stored++;
      }

      // console.error(`   ✅ Stored ${stored} entry points (${skipped} skipped)`);
    } // else {
    // console.error(`   ⚠️  No entry points to store (not an array or undefined)`);
    // }

    // Store key directories (filter out invalid ones)
    if (codebaseAnalysis.keyDirectories && Array.isArray(codebaseAnalysis.keyDirectories)) {
      // let stored = 0;
      // let skipped = 0;

      for (const directory of codebaseAnalysis.keyDirectories) {
        // Skip directories with missing required fields
        if (!directory.path || !directory.type) {
          // skipped++;
          // console.error(`   ⚠️  Skipping invalid directory: path=${directory.path}, type=${directory.type}`);
          continue;
        }

        await storage.insertKeyDirectory({
          id: nanoid(),
          projectPath,
          directoryPath: directory.path,
          directoryType: directory.type,
          fileCount: directory.fileCount || 0,
          description: directory.description
        });
        // stored++;
      }

      // console.error(`   ✅ Stored ${stored} key directories (${skipped} skipped)\n`);
    } // else {
    // console.error(`   ⚠️  No key directories to store (not an array or undefined)\n`);
    // }
  }

  private static inferArchitecturePattern(codebaseAnalysis: any): string {
    const frameworks = codebaseAnalysis?.frameworks || [];
    const directories = codebaseAnalysis?.keyDirectories || [];

    if (frameworks.some((f: string) => f.toLowerCase().includes('react'))) {
      return 'Component-Based (React)';
    } else if (frameworks.some((f: string) => f.toLowerCase().includes('express'))) {
      return 'REST API (Express)';
    } else if (frameworks.some((f: string) => f.toLowerCase().includes('fastapi'))) {
      return 'REST API (FastAPI)';
    } else if (directories.some((d: any) => d.type === 'services')) {
      return 'Service-Oriented';
    } else if (directories.some((d: any) => d.type === 'components')) {
      return 'Component-Based';
    } else if (directories.some((d: any) => d.type === 'models' && d.type === 'views')) {
      return 'MVC Pattern';
    } else {
      return 'Modular';
    }
  }
}
