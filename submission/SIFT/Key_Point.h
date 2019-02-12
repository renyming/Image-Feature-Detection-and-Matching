#pragma once
class Key_Point {
public:
	Key_Point(float m=0, float angle=0):m(m),angle(angle){}
	const float getM() { return m; }
	const float getAngle() { return angle; }

private:
	float m;
	float angle;
};