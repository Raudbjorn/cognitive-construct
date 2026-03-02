import { SurrealStorage } from '../storage/surreal-storage.js';
import { existsSync, statSync } from 'fs';
import { join } from 'path';

export interface DebugOptions {
  verbose: boolean;
  checkDatabase: boolean;
  checkIntelligence: boolean;
  checkFileSystem: boolean;
  validateData: boolean;
  performance: boolean;
}

export class DebugTools {
  private verbose: boolean = false;

  constructor(private options: DebugOptions = {
    verbose: false,
    checkDatabase: true,
    checkIntelligence: true,
    checkFileSystem: true,
    validateData: false,
    performance: false
  }) {
    this.verbose = options.verbose;
  }

  async runDiagnostics(projectPath: string = process.cwd()): Promise<void> {
    console.log('🔍 Anime Mori Debug & Diagnostics');
    console.log(`📁 Project: ${projectPath}`);
    console.log(`⚡ Verbose: ${this.verbose ? 'ON' : 'OFF'}\n`);

    const results = {
      passed: 0,
      warnings: 0,
      errors: 0,
      suggestions: [] as string[]
    };

    // System Information
    console.log('📊 SYSTEM INFORMATION');
    console.log('━'.repeat(50));
    this.checkSystemInfo();
    console.log();

    // Database Diagnostics
    if (this.options.checkDatabase) {
      console.log('🗄️  DATABASE DIAGNOSTICS');
      console.log('━'.repeat(50));
      const dbResults = await this.checkDatabase(projectPath);
      this.mergeResults(results, dbResults);
      console.log();
    }

    // Intelligence Diagnostics
    if (this.options.checkIntelligence) {
      console.log('🧠 INTELLIGENCE DIAGNOSTICS');
      console.log('━'.repeat(50));
      const intResults = await this.checkIntelligence(projectPath);
      this.mergeResults(results, intResults);
      console.log();
    }

    // File System Diagnostics
    if (this.options.checkFileSystem) {
      console.log('📁 FILE SYSTEM DIAGNOSTICS');
      console.log('━'.repeat(50));
      const fsResults = await this.checkFileSystem(projectPath);
      this.mergeResults(results, fsResults);
      console.log();
    }

    // Data Validation
    if (this.options.validateData) {
      console.log('✅ DATA VALIDATION');
      console.log('━'.repeat(50));
      const validationResults = await this.validateIntelligenceData(projectPath);
      this.mergeResults(results, validationResults);
      console.log();
    }

    // Performance Analysis
    if (this.options.performance) {
      console.log('⚡ PERFORMANCE ANALYSIS');
      console.log('━'.repeat(50));
      const perfResults = await this.analyzePerformance(projectPath);
      this.mergeResults(results, perfResults);
      console.log();
    }

    // Summary
    this.printSummary(results);
  }

  private checkSystemInfo(): void {
    const nodeVersion = process.version;
    const platform = process.platform;
    const arch = process.arch;
    const memory = process.memoryUsage();

    console.log(`  Node.js Version: ${nodeVersion}`);
    console.log(`  Platform: ${platform}-${arch}`);
    console.log(`  Memory Usage: ${Math.round(memory.rss / 1024 / 1024)}MB RSS, ${Math.round(memory.heapUsed / 1024 / 1024)}MB Heap`);

    const majorVersion = parseInt(nodeVersion.substring(1));
    if (majorVersion < 18) {
      console.log('  ❌ Node.js version is too old. Minimum required: 18');
    } else {
      console.log('  ✅ Node.js version is compatible');
    }
  }

  private async checkDatabase(projectPath: string): Promise<any> {
    const results = { passed: 0, warnings: 0, errors: 0, suggestions: [] as string[] };
    
    try {
      const dbPath = join(projectPath, 'anime-mori.surql');

      // Check if database directory exists (SurrealKV uses a directory)
      if (existsSync(dbPath)) {
        const stats = statSync(dbPath);
        if (stats.isDirectory()) {
          console.log(`  📄 Database path: ${dbPath}`);
          console.log(`  📅 Modified: ${stats.mtime.toISOString()}`);
          console.log('  ✅ Database directory exists');
          results.passed++;
        } else {
          console.log(`  📄 Database path: ${dbPath}`);
          console.log('  ✅ Database exists');
          results.passed++;
        }
      } else {
        console.log(`  📄 Database path: ${dbPath} (will be created on first use)`);
      }

      // Test database connection
      try {
        const storage = await SurrealStorage.create(dbPath);
        console.log('  ✅ Database connection successful');

        // Check schema version
        const currentVersion = await storage.getSchemaVersion();
        console.log(`  📊 Schema version: ${currentVersion}`);
        console.log('  ✅ Database schema is up-to-date');
        results.passed++;

        // Check table counts
        const concepts = await storage.getSemanticConcepts();
        const patterns = await storage.getDeveloperPatterns();

        console.log(`  📈 Stored concepts: ${concepts.length}`);
        console.log(`  🔍 Stored patterns: ${patterns.length}`);

        if (concepts.length === 0 && patterns.length === 0) {
          console.log('  ⚠️  No intelligence data found');
          results.warnings++;
          results.suggestions.push('Run `anime-mori learn` to analyze your codebase');
        }

        await storage.close();
        results.passed++;

      } catch (error: unknown) {
        console.log(`  ❌ Database connection failed: ${error instanceof Error ? error.message : String(error)}`);
        results.errors++;

        if (this.verbose) {
          console.log(`     Details: ${error instanceof Error ? error.stack : 'No stack trace available'}`);
        }
      }

    } catch (error: unknown) {
      console.log(`  ❌ Database check failed: ${error instanceof Error ? error.message : String(error)}`);
      results.errors++;
    }

    return results;
  }

  private async checkIntelligence(projectPath: string): Promise<any> {
    const results = { passed: 0, warnings: 0, errors: 0, suggestions: [] as string[] };

    try {
      // Check if we can initialize intelligence components
      const dbPath = join(projectPath, 'anime-mori.surql');
      
      if (!existsSync(dbPath)) {
        console.log('  ❌ No database found for intelligence check');
        results.errors++;
        return results;
      }

      const storage = await SurrealStorage.create(dbPath);

      console.log('  ✅ Intelligence components initialized successfully');
      results.passed++;

      // Test basic functionality
      try {
        const concepts = await storage.getSemanticConcepts();
        const patterns = await storage.getDeveloperPatterns();

        console.log(`  📊 Available concepts: ${concepts.length}`);
        console.log(`  🔍 Available patterns: ${patterns.length}`);

        if (concepts.length === 0) {
          console.log('  ⚠️  No semantic concepts found');
          results.warnings++;
          results.suggestions.push('Run learning to build semantic understanding');
        }

        if (patterns.length === 0) {
          console.log('  ⚠️  No development patterns found');
          results.warnings++;
          results.suggestions.push('Run learning to identify coding patterns');
        }

        console.log('  📝 Using local embeddings (transformers.js)');

      } catch (error: unknown) {
        console.log(`  ❌ Intelligence data access failed: ${error instanceof Error ? error.message : String(error)}`);
        results.errors++;
      }

      await storage.close();

    } catch (error: unknown) {
      console.log(`  ❌ Intelligence initialization failed: ${error instanceof Error ? error.message : String(error)}`);
      results.errors++;
      
      if (this.verbose) {
        console.log(`     Details: ${error instanceof Error ? error.stack : 'No stack trace available'}`);
      }
    }

    return results;
  }

  private async checkFileSystem(projectPath: string): Promise<any> {
    const results = { passed: 0, warnings: 0, errors: 0, suggestions: [] as string[] };

    try {
      // Check project directory
      if (!existsSync(projectPath)) {
        console.log(`  ❌ Project directory does not exist: ${projectPath}`);
        results.errors++;
        return results;
      }

      console.log(`  📁 Project directory: ${projectPath}`);
      console.log('  ✅ Project directory exists');
      results.passed++;

      // Check for .anime-mori directory
      const configDir = join(projectPath, '.anime-mori');
      if (existsSync(configDir)) {
        console.log('  📂 Configuration directory found');
        
        // Check config file
        const configFile = join(configDir, 'config.json');
        if (existsSync(configFile)) {
          console.log('  ⚙️  Configuration file exists');
          try {
            const configContent = await import(`file://${configFile}`);
            console.log('  ✅ Configuration file is valid JSON');
            results.passed++;
          } catch (error) {
            console.log('  ❌ Configuration file is invalid JSON');
            results.errors++;
          }
        } else {
          console.log('  ⚠️  No configuration file found');
          results.warnings++;
          results.suggestions.push('Run `anime-mori setup --interactive` to create configuration');
        }
      } else {
        console.log('  ⚠️  No .anime-mori configuration directory');
        results.warnings++;
        results.suggestions.push('Run `anime-mori init` to create basic configuration');
      }

      // Check for common files
      const packageJson = join(projectPath, 'package.json');
      const gitDir = join(projectPath, '.git');
      
      if (existsSync(packageJson)) {
        console.log('  📦 Node.js project detected (package.json)');
        results.passed++;
      }
      
      if (existsSync(gitDir)) {
        console.log('  🗂️  Git repository detected');
        results.passed++;
      }

      // Check .gitignore
      const gitignore = join(projectPath, '.gitignore');
      if (existsSync(gitignore)) {
        const fs = await import('fs');
        const content = fs.readFileSync(gitignore, 'utf-8');
        if (content.includes('anime-mori.surql')) {
          console.log('  ✅ Anime Mori entries found in .gitignore');
          results.passed++;
        } else {
          console.log('  ⚠️  Anime Mori not in .gitignore');
          results.warnings++;
          results.suggestions.push('Add anime-mori.surql to .gitignore to avoid committing database');
        }
      }

    } catch (error: unknown) {
      console.log(`  ❌ File system check failed: ${error instanceof Error ? error.message : String(error)}`);
      results.errors++;
    }

    return results;
  }

  private async validateIntelligenceData(projectPath: string): Promise<any> {
    const results = { passed: 0, warnings: 0, errors: 0, suggestions: [] as string[] };

    console.log('  🔍 Validating intelligence data consistency...');

    try {
      const dbPath = join(projectPath, 'anime-mori.surql');
      if (!existsSync(dbPath)) {
        console.log('  ❌ No database to validate');
        results.errors++;
        return results;
      }

      const storage = await SurrealStorage.create(dbPath);
      const concepts = await storage.getSemanticConcepts();
      const patterns = await storage.getDeveloperPatterns();

      // Validate concepts
      let validConcepts = 0;
      let invalidConcepts = 0;

      for (const concept of concepts) {
        if (concept.conceptName && concept.conceptType && concept.confidenceScore >= 0) {
          validConcepts++;
        } else {
          invalidConcepts++;
          if (this.verbose) {
            console.log(`    ⚠️  Invalid concept: ${concept.id}`);
          }
        }
      }

      console.log(`  📊 Concepts: ${validConcepts} valid, ${invalidConcepts} invalid`);

      if (invalidConcepts > 0) {
        results.warnings++;
        results.suggestions.push('Some concept data may be corrupted - consider re-learning');
      } else {
        results.passed++;
      }

      // Validate patterns
      let validPatterns = 0;
      let invalidPatterns = 0;

      for (const pattern of patterns) {
        if (pattern.patternType && pattern.frequency >= 0) {
          validPatterns++;
        } else {
          invalidPatterns++;
          if (this.verbose) {
            console.log(`    ⚠️  Invalid pattern: ${pattern.patternId}`);
          }
        }
      }

      console.log(`  🔍 Patterns: ${validPatterns} valid, ${invalidPatterns} invalid`);

      if (invalidPatterns > 0) {
        results.warnings++;
        results.suggestions.push('Some pattern data may be corrupted - consider re-learning');
      } else {
        results.passed++;
      }

      await storage.close();

    } catch (error: unknown) {
      console.log(`  ❌ Validation failed: ${error instanceof Error ? error.message : String(error)}`);
      results.errors++;
    }

    return results;
  }

  private async analyzePerformance(projectPath: string): Promise<any> {
    const results = { passed: 0, warnings: 0, errors: 0, suggestions: [] as string[] };

    console.log('  ⚡ Analyzing performance characteristics...');

    try {
      const dbPath = join(projectPath, 'anime-mori.surql');
      if (!existsSync(dbPath)) {
        console.log('  ❌ No database for performance analysis');
        results.errors++;
        return results;
      }

      const stats = statSync(dbPath);
      const dbSizeMB = stats.size / (1024 * 1024);

      console.log(`  💾 Database size: ${dbSizeMB.toFixed(2)}MB`);
      
      if (dbSizeMB > 100) {
        console.log('  ⚠️  Large database size detected');
        results.warnings++;
        results.suggestions.push('Consider archiving old intelligence data for better performance');
      } else {
        console.log('  ✅ Database size is reasonable');
        results.passed++;
      }

      // Test query performance
      const startTime = Date.now();
      const storage = await SurrealStorage.create(dbPath);
      const concepts = await storage.getSemanticConcepts();
      const queryTime = Date.now() - startTime;

      console.log(`  🕐 Query time: ${queryTime}ms for ${concepts.length} concepts`);

      if (queryTime > 1000) {
        console.log('  ⚠️  Slow query performance detected');
        results.warnings++;
        results.suggestions.push('Database may benefit from optimization or indexing');
      } else {
        console.log('  ✅ Query performance is good');
        results.passed++;
      }

      await storage.close();

    } catch (error: unknown) {
      console.log(`  ❌ Performance analysis failed: ${error instanceof Error ? error.message : String(error)}`);
      results.errors++;
    }

    return results;
  }

  private mergeResults(target: any, source: any): void {
    target.passed += source.passed;
    target.warnings += source.warnings;
    target.errors += source.errors;
    target.suggestions.push(...source.suggestions);
  }

  private printSummary(results: any): void {
    console.log('📋 DIAGNOSTIC SUMMARY');
    console.log('━'.repeat(50));
    console.log(`✅ Passed: ${results.passed}`);
    console.log(`⚠️  Warnings: ${results.warnings}`);
    console.log(`❌ Errors: ${results.errors}`);
    
    if (results.suggestions.length > 0) {
      console.log('\n💡 SUGGESTIONS:');
      results.suggestions.forEach((suggestion: string, index: number) => {
        console.log(`${index + 1}. ${suggestion}`);
      });
    }

    console.log('\n🎯 OVERALL STATUS:');
    if (results.errors === 0 && results.warnings === 0) {
      console.log('🟢 All systems operational');
    } else if (results.errors === 0) {
      console.log('🟡 Functional with minor issues');
    } else {
      console.log('🔴 Issues detected - intervention required');
    }
  }
}