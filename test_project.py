#!/usr/bin/env python3
"""
Test script to verify project structure and basic functionality
"""

import os
import sys
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_project_structure():
    """Test if all required files and directories exist"""
    logger.info("Testing project structure...")
    
    required_files = [
        "config.yaml",
        "requirements.txt",
        "main.py",
        "data_preprocessing.py",
        "model_training.py",
        "model_quantization.py",
        "raspberry_pi_deployment.py",
        "README.md",
        ".gitignore"
    ]
    
    required_dirs = [
        "models",
        "logs",
        "checkpoints",
        "deployment",
        "results",
        "test_images"
    ]
    
    missing_files = []
    missing_dirs = []
    
    # Check files
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            logger.info(f"✅ {file}")
    
    # Check directories
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
        else:
            logger.info(f"✅ {directory}/")
    
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")
    
    if missing_dirs:
        logger.warning(f"Missing directories: {missing_dirs}")
    
    return len(missing_files) == 0 and len(missing_dirs) == 0

def test_config_file():
    """Test if configuration file is valid"""
    logger.info("Testing configuration file...")
    
    try:
        with open("config.yaml", 'r') as file:
            config = yaml.safe_load(file)
        
        required_sections = ['data', 'model', 'training', 'quantization', 'deployment', 'security']
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
            else:
                logger.info(f"✅ {section} configuration")
        
        if missing_sections:
            logger.warning(f"Missing configuration sections: {missing_sections}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration file error: {str(e)}")
        return False

def test_imports():
    """Test if all modules can be imported"""
    logger.info("Testing module imports...")
    
    modules = [
        "data_preprocessing",
        "model_training", 
        "model_quantization",
        "raspberry_pi_deployment"
    ]
    
    failed_imports = []
    
    for module in modules:
        try:
            __import__(module)
            logger.info(f"✅ {module}")
        except ImportError as e:
            logger.warning(f"❌ {module}: {str(e)}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_requirements():
    """Test if requirements file exists and has content"""
    logger.info("Testing requirements file...")
    
    if not os.path.exists("requirements.txt"):
        logger.error("❌ requirements.txt not found")
        return False
    
    with open("requirements.txt", 'r') as f:
        content = f.read().strip()
    
    if not content:
        logger.error("❌ requirements.txt is empty")
        return False
    
    required_packages = [
        "tensorflow",
        "numpy",
        "opencv-python",
        "matplotlib"
    ]
    
    missing_packages = []
    for package in required_packages:
        if package not in content:
            missing_packages.append(package)
        else:
            logger.info(f"✅ {package}")
    
    if missing_packages:
        logger.warning(f"Missing packages in requirements.txt: {missing_packages}")
    
    return len(missing_packages) == 0

def main():
    """Run all tests"""
    logger.info("=== Project Test Suite ===")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Configuration File", test_config_file),
        ("Module Imports", test_imports),
        ("Requirements File", test_requirements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with error: {str(e)}")
    
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Project is ready to use.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 