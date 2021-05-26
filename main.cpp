#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include "math/vec2.h"
#include "math/vec3.h"
#include "math/mat4.h"
static float PI = 3.141592;
using filament::math::vec2;
using filament::math::int2;
using filament::math::int3;
using filament::math::uint4;
using filament::math::uint3;
using filament::math::float3;
using filament::math::float4;
using filament::math::mat4f;
using filament::math::mat4;
using filament::math::mat3;
using filament::math::mat3f;
using filament::math::quatf;

inline void decomposeMatrix(const filament::math::mat4f& mat, filament::math::float3* translation,
                            filament::math::quatf* rotation, filament::math::float3* scale) {
  using namespace filament::math;

  // Extract translation.
  *translation = mat[3].xyz;

  // Extract upper-left for determinant computation.
  const float a = mat[0][0];
  const float b = mat[0][1];
  const float c = mat[0][2];
  const float d = mat[1][0];
  const float e = mat[1][1];
  const float f = mat[1][2];
  const float g = mat[2][0];
  const float h = mat[2][1];
  const float i = mat[2][2];
  const float A = e * i - f * h;
  const float B = f * g - d * i;
  const float C = d * h - e * g;

  // Extract scale.
  const float det(a * A + b * B + c * C);
  float scalex = length(float3({a, b, c}));
  float scaley = length(float3({d, e, f}));
  float scalez = length(float3({g, h, i}));
  float3 s = { scalex, scaley, scalez };
  if (det < 0) {
    s = -s;
  }
  *scale = s;

  // Remove scale from the matrix if it is not close to zero.
  mat4f clone = mat;
  if (std::abs(det) > std::numeric_limits<float>::epsilon()) {
    clone[0] /= s.x;
    clone[1] /= s.y;
    clone[2] /= s.z;
    // Extract rotation
    *rotation = clone.toQuaternion();
  } else {
    // Set to identity if close to zero
    *rotation = quatf(1);
  }
}

class Drawer {
public:
  cv::Mat* image_ = nullptr;
  int width_;
  int height_;
  Drawer(int width, int heigth) {
    width_ = width;
    height_ = heigth;
    image_ = new cv::Mat(heigth, width, CV_8UC3, cv::Scalar(255,255,255));
  }

  void clear() {
    delete image_;
    image_ = new cv::Mat(height_, width_, CV_8UC3, cv::Scalar(255,255,255));
  }

  void drawPointAt(int x, int y, int radius, int3 color){
    cv::Point p3(x, y);
    circle(*image_, p3,radius,cv::Scalar(color.z,color.y,color.x),-1);
  }

  void drawLine(int2 from, int2 to, int3 color, int thickness){
    cv::line(*image_, cv::Point(from.x, from.y), cv::Point(to.x, to.y), cv::Scalar(color.z,color.y,color.x), thickness);

  }
};

class Mesh {
public:
  std::vector<float3> points_;
  std::vector<float3> cur_points_;
  mat4f transform_ = mat4f();
  // point index -> affect joint indices
  std::map<int, std::vector<int>> point_to_joints_indices_;
  // point index -> affect joints weights
  std::map<int, std::vector<float>> point_to_joints_weights_;
};

class Joint {
public:
  Joint* parent_ = nullptr;
  std::vector<Joint *> childs_;
  mat4f inverse_binding_matrix_;
  mat4f initial_local_transform_;
  mat4f initial_world_transform_;
  mat4f cur_local_transform_;
  // from joint local -> world
  mat4f currentGlobalJointTransform() const {
    if (parent_) {
      return parent_->currentGlobalJointTransform() * cur_local_transform_;
    } else {
      return cur_local_transform_;
    }
  }
};

class Skin {
public:
  std::vector<Joint *> joints_;
};

void animateJointBlue(Skin* skin, float progress) {
  float3 scale;
  quatf rotation;
  float3 translation;
  Joint* blue_joint = skin->joints_[1];
  decomposeMatrix(blue_joint->initial_local_transform_, &translation, &rotation, &scale);
  blue_joint->cur_local_transform_ = mat4f::translation(translation) *
                                     mat4f::rotation(-45.0 / 180.0 * PI * sin(progress), float3{0,0,1}) * mat4f(rotation) *
                                     mat4f::scaling(scale);
}

void animateJointPurple(Skin* skin, float progress) {
  Joint* purple_joint = skin->joints_[3];
  float3 scale;
  quatf rotation;
  float3 translation;
  decomposeMatrix(purple_joint->initial_local_transform_, &translation, &rotation, &scale);
  purple_joint->cur_local_transform_ = mat4f::translation(translation) *
                                       mat4f::rotation(0.5 * PI * sin(progress) - 0.25 * PI, float3{0,0,1}) * mat4f(rotation) *
                                       mat4f::scaling(scale);
}

void doSkin(Mesh* mesh, Skin* skin) {
  for (int point_idx = 0; point_idx < mesh->points_.size(); point_idx++) {
    if (mesh->point_to_joints_indices_.find(point_idx) == mesh->point_to_joints_indices_.end()) {
      continue;
    }
    auto& point = mesh->points_[point_idx];
    auto& affect_joints_indices = mesh->point_to_joints_indices_.at(point_idx);
    auto& affect_joints_weights = mesh->point_to_joints_weights_.at(point_idx);
    float3 final_point {0,0,0};
    for (int i = 0; i < affect_joints_indices.size(); i++) {
      int affect_joint_idx = affect_joints_indices[i];
      auto joint = skin->joints_[affect_joint_idx];
      auto weight = affect_joints_weights[i];
      auto new_point = (inverse(mesh->transform_)
                       * joint->currentGlobalJointTransform()
                       * joint->inverse_binding_matrix_ * float4 {point.x, point.y, point.z, 1}).xyz;
      final_point += new_point * weight;
    }
    mesh->cur_points_[point_idx] = final_point;
  }
}

void drawJoint(Drawer* drawer, Skin* skin){
  auto& joints = skin->joints_;
  float3 red_joint_position = (joints[0]->currentGlobalJointTransform() * float4{0,0,0,1}).xyz;
  float3 blue_joint_position = (joints[1]->currentGlobalJointTransform() * float4{0,0,0,1}).xyz;
  float3 brown_joint_position = (joints[2]->currentGlobalJointTransform() * float4{0,0,0,1}).xyz;
  float3 purple_joint_position = (joints[3]->currentGlobalJointTransform() * float4{0,0,0,1}).xyz;

  drawer->drawLine(int2{red_joint_position.x, red_joint_position.y}, int2{blue_joint_position.x, blue_joint_position.y}, int3{0,0,0}, 2);
  drawer->drawLine(int2{blue_joint_position.x, blue_joint_position.y}, int2{brown_joint_position.x, brown_joint_position.y}, int3{0,0,0}, 2);
  drawer->drawLine(int2{blue_joint_position.x, blue_joint_position.y}, int2{purple_joint_position.x, purple_joint_position.y}, int3{0,0,0}, 2);

  drawer->drawPointAt(red_joint_position.x, red_joint_position.y, 10, int3{255,0,0});
  drawer->drawPointAt(blue_joint_position.x, blue_joint_position.y, 10, int3{0,0,255});
  drawer->drawPointAt(brown_joint_position.x, brown_joint_position.y, 10, int3{0xd4,0x7f,0});
  drawer->drawPointAt(purple_joint_position.x, purple_joint_position.y, 10, int3{0xf0,0,0xf0});
}

void drawMesh(Drawer* drawer, Mesh* mesh) {
  for (int i = 0; i < mesh->cur_points_.size(); ++i) {
    float3 cur_p = (mesh->transform_ * float4 {mesh->cur_points_[i].x, mesh->cur_points_[i].y, mesh->cur_points_[i].z, 1}).xyz;
    drawer->drawPointAt(cur_p.x, cur_p.y, 10, int3{0,0,0});
    if (i < mesh->cur_points_.size() - 1) {
      float3 next_p = (mesh->transform_ * float4 {mesh->cur_points_[i+1].x, mesh->cur_points_[i+1].y, mesh->cur_points_[i+1].z, 1}).xyz;
      drawer->drawLine(int2{cur_p.x, cur_p.y}, int2{next_p.x, next_p.y}, int3{0,0,0}, 2);
    }
  }
}

int main(int argc, const char** argv)
{
  // ----------------- skin
  Joint* red_joint = new Joint();
  Joint* blue_joint = new Joint();
  Joint* brown_joint = new Joint();
  Joint* purple_joint = new Joint();

  red_joint->childs_.push_back(blue_joint);
  blue_joint->parent_ = red_joint;

  blue_joint->childs_.push_back(purple_joint);
  purple_joint->parent_ = blue_joint;

  blue_joint->childs_.push_back(brown_joint);
  brown_joint->parent_ = blue_joint;

  float4 p;
  red_joint->initial_world_transform_ = mat4f::translation(float3{114.63, 74, 0});
  red_joint->initial_local_transform_ = red_joint->initial_world_transform_;
  red_joint->cur_local_transform_ = red_joint->initial_local_transform_;

  blue_joint->initial_world_transform_ = mat4f::translation(float3{194.64, 155, 0});
  blue_joint->initial_local_transform_ = inverse(red_joint->initial_world_transform_) * blue_joint->initial_world_transform_;
  blue_joint->cur_local_transform_ = blue_joint->initial_local_transform_;

  brown_joint->initial_world_transform_ = mat4f::translation(float3{173.57, 295, 0});
  brown_joint->initial_local_transform_ = inverse(blue_joint->initial_world_transform_) * brown_joint->initial_world_transform_;
  brown_joint->cur_local_transform_ = brown_joint->initial_local_transform_;

  purple_joint->initial_world_transform_ = mat4f::translation(float3{214.57, 295, 0});
  purple_joint->initial_local_transform_ = inverse(blue_joint->initial_world_transform_) * purple_joint->initial_world_transform_;
  purple_joint->cur_local_transform_ = purple_joint->initial_local_transform_;

  Skin* skin = new Skin();
  skin->joints_ = {red_joint, blue_joint, brown_joint, purple_joint};

  // mesh
  Mesh* mesh = new Mesh();
  mesh->points_ = {
      float3 {154.64, 74, 0},
      float3 {195.64, 114, 0},
      float3 {235.64, 154, 0},
      float3 {235.70, 214.03, 0},
      float3 {235.70, 276.03, 0},
      float3 {274.64, 315, 0},
      float3 {313.64, 355, 0},
      float3 {273.64, 395, 0},
      float3 {233.64, 356, 0},
      float3 {194.64, 316, 0},
      float3 {154.64, 356, 0},
      float3 {114.63, 396, 0},
      float3 {74.63, 356, 0},
      float3 {114.63, 316, 0},
      float3 {155.64, 275.03, 0},
      float3 {155.64, 214.03, 0},
      float3 {155.64, 155, 0},
      float3 {114.63, 114, 0},
      float3 {74.63, 74, 0},
  };
  mesh->cur_points_ = mesh->points_;

  mesh->point_to_joints_indices_[1] = {0};
  mesh->point_to_joints_weights_[1] = {1.0};
  mesh->point_to_joints_indices_[2] = {0};
  mesh->point_to_joints_weights_[2] = {1.0};
  mesh->point_to_joints_indices_[17] = {0};
  mesh->point_to_joints_weights_[17] = {1.0};
  mesh->point_to_joints_indices_[16] = {0};
  mesh->point_to_joints_weights_[16] = {1.0};

  mesh->point_to_joints_indices_[3] = {1};
  mesh->point_to_joints_weights_[3] = {1.0};
  mesh->point_to_joints_indices_[4] = {1};
  mesh->point_to_joints_weights_[4] = {1.0};
  mesh->point_to_joints_indices_[9] = {1};
  mesh->point_to_joints_weights_[9] = {1.0};
  mesh->point_to_joints_indices_[14] = {1};
  mesh->point_to_joints_weights_[14] = {1.0};
  mesh->point_to_joints_indices_[15] = {1};
  mesh->point_to_joints_weights_[15] = {1.0};

  mesh->point_to_joints_indices_[5] = {3};
  mesh->point_to_joints_weights_[5] = {1.0};
  mesh->point_to_joints_indices_[6] = {3};
  mesh->point_to_joints_weights_[6] = {1.0};
  mesh->point_to_joints_indices_[7] = {3};
  mesh->point_to_joints_weights_[7] = {1.0};
  mesh->point_to_joints_indices_[8] = {3};
  mesh->point_to_joints_weights_[8] = {1.0};

  mesh->point_to_joints_indices_[10] = {2};
  mesh->point_to_joints_weights_[10] = {1.0};
  mesh->point_to_joints_indices_[11] = {2};
  mesh->point_to_joints_weights_[11] = {1.0};
  mesh->point_to_joints_indices_[12] = {2};
  mesh->point_to_joints_weights_[12] = {1.0};
  mesh->point_to_joints_indices_[13] = {2};
  mesh->point_to_joints_weights_[13] = {1.0};

  // build joint inverse bind matrix
  red_joint->inverse_binding_matrix_ = inverse(red_joint->initial_local_transform_)
                                       * mesh->transform_;

  blue_joint->inverse_binding_matrix_ = inverse(blue_joint->initial_local_transform_)
                                        * inverse(red_joint->initial_local_transform_)
                                        * mesh->transform_;

  brown_joint->inverse_binding_matrix_ = inverse(brown_joint->initial_local_transform_)
                                         * inverse(blue_joint->initial_local_transform_)
                                         * inverse(red_joint->initial_local_transform_)
                                         * mesh->transform_;
  purple_joint->inverse_binding_matrix_ = inverse(purple_joint->initial_local_transform_)
                                         * inverse(blue_joint->initial_local_transform_)
                                         * inverse(red_joint->initial_local_transform_)
                                         * mesh->transform_;

  int width = 800;
  int height = 800;

  auto drawer = new Drawer(width, height);
  int key = 0;

  float progress_blue = 0;
  float progress_purple = 0;
  float step_blue = 0.02;
  float step_purple = 0.06;
  while (key != 27) {
    progress_blue += step_blue;
    if (progress_blue > 1 || progress_blue < 0) {
      step_blue *= -1;
    }
    progress_purple += step_purple;
    if (progress_purple > 1 || progress_purple < 0) {
      step_purple *= -1;
    }
    animateJointBlue(skin, progress_blue);
    animateJointPurple(skin, progress_purple);
    doSkin(mesh, skin);

    drawer->clear();
    drawJoint(drawer, skin);
    drawMesh(drawer, mesh);

    cv::imshow("image", *drawer->image_);
    key = cv::waitKey(15);
  }

    return 0;
}
