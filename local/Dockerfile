FROM duckietown/rpi-duckiebot-base:master19

RUN ["cross-build-start"]

#### START CUSTOM CATKIN_WS ####
RUN /bin/bash -c "mkdir -p custom_ws/src/"

# Copy or init your packages in here
COPY dt_dependent_node custom_ws/src/dt_dependent_node
RUN chmod +x custom_ws/src/dt_dependent_node/dt_dependent_node.py

# Here's a copied package, slighly edited. We don't launch this, but you could easily 
# launch this by finding `custom_line_detector`# and putting that in the 
# lf_slim.launch file  
COPY custom_line_detector custom_ws/src/custom_line_detector
# Don't forget to "chmod" any node you may need inside!

RUN /bin/bash -c "cd custom_ws/src/"

# Do not change the below line! This ensures that your workspace is 
# overlayed on top of the Duckietown stack!  
# MAKE sure this line is present in the build: 
# This workspace overlays: /home/software/catkin_ws/devel;/opt/ros/kinetic
RUN /bin/bash -c "source /home/software/catkin_ws/devel/setup.bash && catkin_init_workspace && cd ../.."
RUN /bin/bash -c "source /home/software/catkin_ws/devel/setup.bash && catkin_make -j -C custom_ws/"
RUN /bin/bash -c "source custom_ws/devel/setup.bash"

# Copy your launch file over
COPY lf_slim.launch .
#### END CUSTOM CATKIN_WS ####

RUN ["cross-build-end"]
