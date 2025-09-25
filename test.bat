@echo off
echo Running tests...

if not exist bin mkdir bin

REM Correctly compile with the JUnit jar in classpath
javac -cp "lib\junit-platform-console-standalone-1.10.2.jar" -d bin src\core\*.java src\data\*.java src\layers\*.java src\tools\*.java test\*.java

if %errorlevel% neq 0 (
    echo Compilation failed.
    exit /b %errorlevel%
)

echo Build successful.

REM Run tests with JUnit standalone
java -jar lib\junit-platform-console-standalone-1.10.2.jar --class-path bin --scan-class-path

pause
