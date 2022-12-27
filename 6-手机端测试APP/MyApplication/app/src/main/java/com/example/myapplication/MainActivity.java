package com.example.myapplication;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.app.ActivityManager;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.app.AlertDialog.Builder;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.Kwarg;
import com.chaquo.python.PyObject;
import com.chaquo.python.android.AndroidPlatform;
import com.chaquo.python.Python;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    static final String TAG = "PythonOnAndroid";
    private Button mButton;
    private Integer startlayer;  //开始的第一层
    private  Integer endlayer;
    public TextView text;
    public EditText editText;
    public ActivityManager am;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setContentView(R.layout.activity_main);
        initPython();
        //callPythonCode();
        setContentView(R.layout.activity_main);
        mButton = findViewById(R.id.run_button);
        View.OnClickListener onClickListener = new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(TAG, "onClick: ");
                editText =(EditText)findViewById(R.id.inputstart);
                String startlayer_string = editText.getText().toString();
                if(startlayer_string != "") {
                    callPythonCode();
                }
            }
        };
        mButton.setOnClickListener(onClickListener);

    }
    // 初始化Python环境
    void initPython(){
        if (! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
    }
    // 调用python代码
    void callPythonCode(){
        Python py = Python.getInstance();
        Double runtime = 0.0;
        //py.getModule("hello").callAttr("greet", "Android");
        editText =(EditText)findViewById(R.id.inputstart);
        String startlayer_string = editText.getText().toString();
        editText = (EditText)findViewById(R.id.inputend);
        String endlayer_string = editText.getText().toString();
        if(startlayer_string=="" || endlayer_string=="")
        {
            toastMessage("请输入开始层或结束层！");
        }
        else {
            startlayer = Integer.valueOf(startlayer_string);
            endlayer = Integer.valueOf(endlayer_string);
            if (startlayer >= 0 && endlayer <= 12 && startlayer <= endlayer) {
                Log.d(TAG, "startlayer = " + startlayer_string);
                PyObject obj5 = py.getModule("simplecloud").callAttr("alexnet", startlayer, endlayer);
                runtime = obj5.toJava(Double.class);
                Log.d(TAG, "runtime = " + runtime.toString());
                String output = "设备运行第" + startlayer.toString() + "层到第" + endlayer.toString() + "的时延是：" + runtime.toString() + "s(10张图片)";
                text= (TextView) findViewById(R.id.output);
                text.setText(output);//设置文字内容
                text.setTextSize(8);;//设置字体大小
                Log.e("latency!!!", runtime.toString());
                //Memory
                am = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
                ActivityManager.MemoryInfo outInfo = new ActivityManager.MemoryInfo();
                am.getMemoryInfo(outInfo);
                double memory = outInfo.availMem/1000.0/1000.0/1000.0;
                Log.e("memory!!!", String.valueOf(memory));
            } else {
                toastMessage("请检查开始层的规格是否正确！");
            }
        }
        while (true) {
            PyObject obj5 = py.getModule("simplecloud_googlenet").callAttr("GoogleNet", startlayer, endlayer);
            runtime = obj5.toJava(Double.class);
            am = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
            ActivityManager.MemoryInfo outInfo = new ActivityManager.MemoryInfo();
            am.getMemoryInfo(outInfo);
            double memory = outInfo.availMem/1000.0/1000.0/1000.0;
            Log.e("latency!!!", runtime.toString());
            Log.e("memory!!!", String.valueOf(memory));
            try {
                Thread.sleep(5000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        //memory test
        //ActivityManager am;
        //am = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
        //ActivityManager.MemoryInfo outInfo = new ActivityManager.MemoryInfo();
        //am.getMemoryInfo(outInfo);
        //double memory = outInfo.availMem/1000.0/1000.0/1000.0;
        //Log.e("memory!!!", String.valueOf(memory));
    }
    public void toastMessage(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
    }

}








