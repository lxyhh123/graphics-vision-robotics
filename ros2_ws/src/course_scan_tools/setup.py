from setuptools import setup

package_name = "course_scan_tools"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    description="LaserScan observation for the course Gazebo lab.",
    entry_points={"console_scripts": ["scan_summary = course_scan_tools.scan_summary:main"]},
)
